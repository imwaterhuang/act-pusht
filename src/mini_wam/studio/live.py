"""Continuous interactive Push-T sessions, separate from scored rollouts."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from PIL import Image

from mini_wam.studio.checkpoints import _load_act, ACTStudioAdapter
from mini_wam.studio.pusht import (
    make_pusht_env, random_scene, validate_scene, prepare_model_input, denormalize_actions,
)


class LiveSession:
    """One session owns its physics state; only the simulation loop mutates it."""

    def __init__(self, model, normalization, seed=0):
        self.model = model
        self.normalization = normalization
        self.device = next(model.parameters()).device
        self.env = make_pusht_env().unwrapped
        self.history = deque(maxlen=2)
        self.actions = deque()
        self.paused = False
        self.drag_target = None
        self.steps = self.model_calls = self.moves = 0
        self.seed = seed
        self.reset(seed)

    def reset(self, seed):
        for candidate in range(seed, seed + 200):
            scene = random_scene(candidate)
            try:
                validate_scene(scene)
            except ValueError:
                continue
            self.seed = candidate
            obs, _ = self.env.reset(seed=candidate, options={"reset_to_state": scene.as_array()})
            self.history.clear()
            self.history.extend([obs, obs])
            self.actions.clear()
            self.drag_target = None
            self.steps = self.model_calls = self.moves = 0
            return
        raise ValueError("无法生成合法场景")

    def move_block(self, x, y):
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("拖动坐标必须为有限数值")
        body = self.env.block
        body.position = (float(np.clip(x, 100, 400)), float(np.clip(y, 100, 400)))
        body.velocity = (0, 0)
        body.angular_velocity = 0
        body.force = (0, 0)
        body.torque = 0
        body.activate()
        self.env.space.reindex_shapes_for_body(body)
        obs = self.env.get_obs()
        self.history.clear()
        self.history.extend([obs, obs])
        # Never finish an action chunk predicted before the intervention.
        self.actions.clear()

    def tick(self, commands):
        if "paused" in commands:
            self.paused = commands["paused"]
        if "reset" in commands:
            self.reset(self.seed + 1)
        if "drag" in commands:
            drag = commands["drag"]
            self.move_block(drag["x"], drag["y"])
            self.moves += 1
            self.drag_target = (drag["x"], drag["y"]) if drag["held"] else None
        if self.drag_target is not None:
            self.move_block(*self.drag_target)
        if not self.paused:
            if not self.actions:
                images, positions = prepare_model_input(list(self.history), self.normalization, self.device)
                with torch.inference_mode():
                    prediction = self.model(images, positions)
                self.actions.extend(denormalize_actions(prediction, self.normalization)[:4])
                self.model_calls += 1
            # The unwrapped environment supports ongoing physics after success.
            # No TimeLimit wrapper and no success-triggered reset in live mode.
            obs, _, _, _, _ = self.env.step(self.actions.popleft())
            self.history.append(obs)
            self.steps += 1
        if self.drag_target is not None:
            self.move_block(*self.drag_target)
        body = self.env.block
        polygons = [
            [[float(v.x), float(v.y)] for v in
             (body.local_to_world(point) for point in shape.get_vertices())]
            for shape in body.shapes
        ]
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(self.env.render())).save(buffer, format="JPEG", quality=85)
        return {
            "type": "frame", "image": base64.b64encode(buffer.getvalue()).decode(),
            "steps": self.steps, "model_calls": self.model_calls, "moves": self.moves,
            "paused": self.paused, "seed": self.seed,
            "coverage": float(self.env._get_coverage()),
            "block": [float(body.position.x), float(body.position.y)], "polygons": polygons,
        }

    def close(self):
        self.env.close()


def register_live_routes(app: FastAPI, checkpoint: Path):
    @app.get("/live", response_class=HTMLResponse)
    def live_page():
        return Path(__file__).with_name("live.html").read_text(encoding="utf-8")

    @app.websocket("/live/ws")
    async def live_socket(ws: WebSocket):
        # Only the local workbench may open a control connection.
        origin = ws.headers.get("origin")
        if origin and origin != f"http://{ws.headers.get('host')}":
            await ws.close(code=1008)
            return
        await ws.accept()
        session = None
        receiver = None
        pending = {}
        disconnected = asyncio.Event()

        async def receive_commands():
            try:
                while True:
                    command = await ws.receive_json()
                    if not isinstance(command, dict):
                        continue
                    if command.get("type") == "drag":
                        try:
                            x, y = float(command["x"]), float(command["y"])
                        except (KeyError, ValueError, TypeError):
                            continue
                        if math.isfinite(x) and math.isfinite(y):
                            pending["drag"] = {"x": x, "y": y, "held": command.get("held") is True}
                    elif command.get("type") == "pause":
                        pending["paused"] = command.get("value") is True
                    elif command.get("type") == "reset":
                        pending["reset"] = True
            except WebSocketDisconnect:
                pass
            finally:
                disconnected.set()

        def create_session():
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model, stats, payload = _load_act(checkpoint, device)
            if payload["step"] != 30000:
                raise ValueError("实时页面要求 30,000 步 ACT 检查点")
            return LiveSession(ACTStudioAdapter(model).eval(), stats)

        try:
            receiver = asyncio.create_task(receive_commands())
            session = await asyncio.to_thread(create_session)
            while not disconnected.is_set():
                started = asyncio.get_running_loop().time()
                commands = dict(pending)
                pending.clear()
                frame = await asyncio.to_thread(session.tick, commands)
                await ws.send_json(frame)
                await asyncio.sleep(max(0, 0.1 - (asyncio.get_running_loop().time() - started)))
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            with contextlib.suppress(Exception):
                await ws.send_json({"type": "error", "message": str(exc)})
        finally:
            if receiver:
                receiver.cancel()
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect):
                    await receiver
            if session:
                session.close()
