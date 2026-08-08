"""PySide6 industrial-pixel desktop interface for the MiniVLA V3 demo."""

from __future__ import annotations

import multiprocessing as mp
from functools import partial
from queue import Empty

import numpy as np
from PySide6.QtCore import (
    QEasingCurve,
    QObject,
    QPropertyAnimation,
    QTimer,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QCloseEvent, QFont, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from scripts.collect_data_v2 import PHASE_NAMES
from utils.instruction_resolver_v3 import (
    InstructionResolutionError,
    ReliableInstructionResolver,
    ResolvedInstruction,
)
from utils.interactive_session_v3 import (
    DEFAULT_DEMO_POLICY,
    DEFAULT_DEMO_SEED,
    demo_process_main,
)


APP_STYLESHEET = """
QWidget {
    background: #0b0f10;
    color: #dfe8e5;
    font-family: Menlo, Monaco, "Courier New", monospace;
    font-size: 13px;
    letter-spacing: 0px;
}
QMainWindow { background: #080c0d; }
QFrame#header, QFrame#footer, QFrame#panel, QFrame#cameraPanel {
    background: #111718;
    border: 1px solid #394745;
    border-radius: 0px;
}
QLabel#appTitle {
    color: #6ee7c4;
    font-size: 23px;
    font-weight: 800;
}
QLabel#appSubtitle, QLabel#sectionLabel, QLabel#cameraMeta {
    color: #81918e;
    font-size: 11px;
    font-weight: 700;
}
QLabel#cameraTitle {
    color: #b8c8c4;
    font-size: 12px;
    font-weight: 800;
}
QLabel#statusBadge {
    background: #17211f;
    border: 1px solid #55726b;
    color: #6ee7c4;
    font-weight: 800;
    padding: 7px 12px;
}
QLabel#statusBadge[state="busy"] {
    color: #f1bd65;
    border-color: #8a6932;
    background: #211b11;
}
QLabel#statusBadge[state="error"] {
    color: #ff7474;
    border-color: #8f4040;
    background: #241313;
}
QLabel#intentLabel {
    color: #6ee7c4;
    font-size: 15px;
    font-weight: 800;
    padding: 7px 0px;
}
QLabel#errorLabel {
    color: #ff7d7d;
    min-height: 34px;
}
QLabel#errorLabel[state="valid"] { color: #7f918d; }
QLabel#valueLabel {
    color: #f0f5f3;
    font-weight: 700;
}
QLabel#metricLabel { color: #758784; }
QFrame#resultOverlay { background: transparent; border: none; }
QLineEdit {
    background: #090d0e;
    border: 2px solid #4c625d;
    color: #f4faf8;
    font-size: 15px;
    padding: 11px;
    selection-background-color: #276b5b;
}
QLineEdit:focus { border-color: #6ee7c4; }
QLineEdit:disabled { color: #66726f; border-color: #293331; }
QPushButton, QToolButton {
    background: #17201f;
    border: 1px solid #50615e;
    color: #dfe8e5;
    min-height: 34px;
    padding: 6px 12px;
    font-weight: 800;
}
QPushButton:hover, QToolButton:hover {
    background: #22302d;
    border-color: #6ee7c4;
    color: #ffffff;
}
QPushButton:pressed, QToolButton:pressed { background: #0d1514; }
QPushButton:disabled, QToolButton:disabled {
    background: #101514;
    border-color: #27302f;
    color: #4d5956;
}
QPushButton#startButton {
    background: #174b40;
    border-color: #6ee7c4;
    color: #eafff9;
}
QPushButton#stopButton { border-color: #985050; }
QPushButton#nudgeButton {
    border-color: #9b7432;
    color: #f1bd65;
}
QToolButton[target="A"] { border-bottom: 3px solid #e85d5d; }
QToolButton[target="B"] { border-bottom: 3px solid #5b9cff; }
QToolButton[target="C"] { border-bottom: 3px solid #59cc7a; }
QFrame#cameraPanel:hover { border-color: #6ee7c4; }
"""


CAMERA_SOURCES = {
    "frontview": ("SIMULATION // FRONTVIEW", "512 x 512 RGB"),
    "agentview": ("AGENT VIEW", "MODEL INPUT // 112 x 112"),
    "wrist": ("WRIST VIEW", "MODEL INPUT // 112 x 112"),
}


class ResultOverlay(QFrame):
    """Paint result text directly, avoiding any opaque label background."""

    COLORS = {
        "success": (QColor("#baffec"), QColor("#6ee7c4"), "TASK COMPLETE"),
        "fail": (QColor("#ff8a8a"), QColor("#ff6666"), "RUN COMPLETE"),
        "stop": (QColor("#f5cf84"), QColor("#f1bd65"), "RUN INTERRUPTED"),
    }

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("resultOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self._text = ""
        self._result = "success"

    def set_result(self, text: str, result: str) -> None:
        self._text = text
        self._result = result
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        del event
        if not self._text:
            return
        text_color, kicker_color, kicker = self.COLORS.get(
            self._result,
            self.COLORS["fail"],
        )
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        kicker_rect = self.rect().adjusted(0, 7, 0, -92)
        result_rect = self.rect().adjusted(0, 32, 0, -8)
        shadow = QColor(0, 0, 0, 220)

        kicker_font = QFont("Menlo")
        kicker_font.setPixelSize(12)
        kicker_font.setWeight(QFont.Weight.Bold)
        painter.setFont(kicker_font)
        painter.setPen(shadow)
        painter.drawText(
            kicker_rect.translated(0, 2),
            Qt.AlignmentFlag.AlignCenter,
            kicker,
        )
        painter.setPen(kicker_color)
        painter.drawText(kicker_rect, Qt.AlignmentFlag.AlignCenter, kicker)

        result_font = QFont("Menlo")
        result_font.setPixelSize(52)
        result_font.setWeight(QFont.Weight.Black)
        painter.setFont(result_font)
        for x_offset, y_offset in ((-2, 3), (0, 4), (2, 3)):
            painter.setPen(shadow)
            painter.drawText(
                result_rect.translated(x_offset, y_offset),
                Qt.AlignmentFlag.AlignCenter,
                self._text,
            )
        painter.setPen(text_color)
        painter.drawText(
            result_rect,
            Qt.AlignmentFlag.AlignCenter,
            self._text,
        )
        painter.end()


class CameraView(QFrame):
    """Stable aspect-ratio RGB image panel with an optional result overlay."""

    clicked = Signal(str)

    def __init__(
        self,
        title: str,
        metadata: str,
        minimum_size: int,
        slot_id: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("cameraPanel")
        self.slot_id = slot_id
        self._image: QImage | None = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click to show this camera in the main monitor")
        self._image_label = QLabel("WAITING FOR VIDEO")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self._image_label.setStyleSheet(
            "background: #030606; color: #41504d; border: 1px solid #283330;"
        )
        self._image_label.setMinimumSize(minimum_size, minimum_size)
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._overlay = ResultOverlay(self._image_label)
        self._overlay_animation = QPropertyAnimation(
            self._overlay,
            b"geometry",
            self,
        )
        self._overlay.hide()

        self._title_label = QLabel(title)
        self._title_label.setObjectName("cameraTitle")
        self._title_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self._metadata_label = QLabel(metadata)
        self._metadata_label.setObjectName("cameraMeta")
        self._metadata_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        header = QHBoxLayout()
        header.addWidget(self._title_label)
        header.addStretch(1)
        header.addWidget(self._metadata_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 9, 10, 10)
        layout.setSpacing(7)
        layout.addLayout(header)
        layout.addWidget(self._image_label, 1)

    def set_source(self, source_key: str) -> None:
        title, metadata = CAMERA_SOURCES[source_key]
        self._title_label.setText(title)
        self._metadata_label.setText(metadata)

    def set_frame(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        rgb = np.ascontiguousarray(frame, dtype=np.uint8)
        height, width, channels = rgb.shape
        if channels != 3:
            raise ValueError(f"Expected RGB image, got {rgb.shape}")
        self._image = QImage(
            rgb.data,
            width,
            height,
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._image is None:
            return
        size = self._image_label.size()
        pixmap = QPixmap.fromImage(self._image).scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._image_label.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._position_overlay()
        self._refresh_pixmap()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.slot_id)
        super().mousePressEvent(event)

    def _position_overlay(self) -> None:
        image_rect = self._image_label.rect()
        width = max(240, int(image_rect.width() * 0.78))
        width = min(width, max(1, image_rect.width() - 24))
        height = min(150, max(96, int(image_rect.height() * 0.28)))
        left = image_rect.x() + (image_rect.width() - width) // 2
        top = image_rect.y() + (image_rect.height() - height) // 2
        self._overlay.setGeometry(left, top, width, height)

    def show_result(
        self,
        text: str,
        result: str,
        animate: bool = False,
    ) -> None:
        self._overlay.set_result(text, result)
        self._position_overlay()
        self._overlay_animation.stop()
        self._overlay.show()
        self._overlay.raise_()
        if animate:
            final_geometry = self._overlay.geometry()
            start_geometry = final_geometry.translated(0, 18)
            self._overlay_animation.setDuration(380)
            self._overlay_animation.setEasingCurve(
                QEasingCurve.Type.OutCubic
            )
            self._overlay_animation.setStartValue(start_geometry)
            self._overlay_animation.setEndValue(final_geometry)
            self._overlay_animation.start()

    def clear_result(self) -> None:
        self._overlay_animation.stop()
        self._overlay.hide()


class DemoProcessController(QObject):
    """Bridge Qt to a spawned MuJoCo process using queues and events."""

    initialized = Signal(object)
    scene_ready = Signal(object)
    episode_ready = Signal(object)
    step_ready = Signal(object)
    episode_finished = Signal(object)
    nudge_status = Signal(object)
    error = Signal(str)
    shutdown_complete = Signal()

    def __init__(
        self,
        policy_path: str,
        seed: int,
        max_steps: int,
        local_files_only: bool,
    ) -> None:
        super().__init__()
        self.policy_path = policy_path
        self.seed = seed
        self.max_steps = max_steps
        self.local_files_only = local_files_only
        context = mp.get_context("spawn")
        self.command_queue = context.Queue()
        self.status_queue = context.Queue()
        self.frame_queue = context.Queue(maxsize=4)
        self.stop_event = context.Event()
        self.nudge_event = context.Event()
        self.process = context.Process(
            target=demo_process_main,
            args=(
                policy_path,
                seed,
                max_steps,
                local_files_only,
                self.command_queue,
                self.status_queue,
                self.frame_queue,
                self.stop_event,
                self.nudge_event,
            ),
            name="minivla-demo-runtime",
            daemon=True,
        )
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(25)
        self.poll_timer.timeout.connect(self._poll)
        self._shutdown_sent = False
        self._shutdown_emitted = False
        self._unexpected_exit_reported = False

    def start(self) -> None:
        self.process.start()
        self.poll_timer.start()

    def start_task(self, resolved: ResolvedInstruction) -> None:
        self.stop_event.clear()
        self.nudge_event.clear()
        self.command_queue.put({"type": "start", "resolved": resolved})

    def reset_scene(self) -> None:
        self.stop_event.clear()
        self.nudge_event.clear()
        self.command_queue.put({"type": "reset"})

    def request_stop(self) -> None:
        self.stop_event.set()

    def request_nudge(self) -> bool:
        if not self.process.is_alive() or self._shutdown_sent:
            return False
        self.nudge_event.set()
        return True

    def shutdown(self) -> None:
        if self._shutdown_sent:
            return
        self._shutdown_sent = True
        self.stop_event.set()
        self.command_queue.put({"type": "shutdown"})
        QTimer.singleShot(8000, self._force_shutdown)

    @Slot()
    def _poll(self) -> None:
        # Frames are emitted first so the final control-step image reaches the
        # UI before an episode-finished status freezes the monitor.
        while True:
            try:
                message = self.frame_queue.get_nowait()
            except Empty:
                break
            if message.get("kind") == "scene":
                self.scene_ready.emit(message["payload"])
            elif message.get("kind") == "step":
                self.step_ready.emit(message["payload"])

        while True:
            try:
                message = self.status_queue.get_nowait()
            except Empty:
                break
            kind = message.get("kind")
            if kind == "initialized":
                self.initialized.emit(message)
            elif kind == "episode_ready":
                self.episode_ready.emit(message["payload"])
            elif kind == "episode_finished":
                self.episode_finished.emit(message["payload"])
            elif kind == "nudge_status":
                self.nudge_status.emit(message)
            elif kind == "error":
                self.error.emit(message["message"])
            elif kind == "shutdown_complete":
                self._finish_shutdown()

        if (
            not self.process.is_alive()
            and not self._shutdown_emitted
            and not self._unexpected_exit_reported
        ):
            self._unexpected_exit_reported = True
            self.error.emit(
                f"Simulation process exited unexpectedly with code "
                f"{self.process.exitcode}"
            )
            self._finish_shutdown()

    def _force_shutdown(self) -> None:
        if self._shutdown_emitted:
            return
        if self.process.is_alive():
            self.process.terminate()
        self._finish_shutdown()

    def _finish_shutdown(self) -> None:
        if self._shutdown_emitted:
            return
        self._shutdown_emitted = True
        self.poll_timer.stop()
        self.process.join(timeout=2.0)
        self.shutdown_complete.emit()


class MiniVLADemoWindow(QMainWindow):
    def __init__(
        self,
        policy_path: str = DEFAULT_DEMO_POLICY,
        seed: int = DEFAULT_DEMO_SEED,
        max_steps: int = 200,
        local_files_only: bool = True,
        start_backend: bool = True,
        close_when_ready: bool = False,
        smoke_command: str | None = None,
        smoke_nudge: bool = False,
    ) -> None:
        super().__init__()
        self.setWindowTitle("MiniVLA Control Deck")
        self.setMinimumSize(1320, 880)
        self.resize(1420, 900)
        self.setStyleSheet(APP_STYLESHEET)
        self.resolver = ReliableInstructionResolver()
        self.resolved_instruction: ResolvedInstruction | None = None
        self._ready = False
        self._running = False
        self._scene_consumed = False
        self._reset_after_run = False
        self._allow_close = False
        self._closing = False
        self._nudge_available = False
        self._video_frozen = False
        self._latest_frames: dict[str, np.ndarray] = {}
        self._camera_slot_sources = {
            "main": "frontview",
            "aux_left": "agentview",
            "aux_right": "wrist",
        }
        self._close_when_ready = close_when_ready
        self._smoke_command = smoke_command
        self._smoke_nudge = smoke_nudge
        self.backend: DemoProcessController | None = None
        self._build_ui()
        self._connect_ui()
        self.command_input.setText("Pick up the red cube")
        if start_backend:
            self._start_backend(policy_path, seed, max_steps, local_files_only)
        else:
            self._set_status("UI PREVIEW", "busy")

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        self.setCentralWidget(central)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 10, 14, 10)
        title_column = QVBoxLayout()
        title_column.setSpacing(1)
        title = QLabel("MINIVLA // CONTROL DECK")
        title.setObjectName("appTitle")
        subtitle = QLabel("V3 POLICY // PANDA OSC_POSE // TEMPORAL ACT")
        subtitle.setObjectName("appSubtitle")
        title_column.addWidget(title)
        title_column.addWidget(subtitle)
        self.status_badge = QLabel("BOOTING SYSTEM")
        self.status_badge.setObjectName("statusBadge")
        self.status_badge.setProperty("state", "busy")
        header_layout.addLayout(title_column)
        header_layout.addStretch(1)
        header_layout.addWidget(self.status_badge)
        root.addWidget(header)

        body = QHBoxLayout()
        body.setSpacing(10)
        self.front_camera = CameraView(
            "SIMULATION // FRONTVIEW",
            "512 x 512 RGB",
            560,
            "main",
        )
        body.addWidget(self.front_camera, 7)

        side = QVBoxLayout()
        side.setSpacing(10)
        camera_row = QHBoxLayout()
        camera_row.setSpacing(10)
        self.agent_camera = CameraView(
            "AGENT VIEW",
            "MODEL INPUT // 112 x 112",
            190,
            "aux_left",
        )
        self.wrist_camera = CameraView(
            "WRIST VIEW",
            "MODEL INPUT // 112 x 112",
            190,
            "aux_right",
        )
        camera_row.addWidget(self.agent_camera, 1)
        camera_row.addWidget(self.wrist_camera, 1)
        side.addLayout(camera_row, 3)

        command_panel = QFrame()
        command_panel.setObjectName("panel")
        command_layout = QVBoxLayout(command_panel)
        command_layout.setContentsMargins(13, 11, 13, 12)
        command_layout.setSpacing(8)
        section = QLabel("COMMAND INPUT")
        section.setObjectName("sectionLabel")
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("e.g. Pick the azure ball up")
        self.intent_label = QLabel("AWAITING COMMAND")
        self.intent_label.setObjectName("intentLabel")
        self.intent_label.setWordWrap(True)
        self.error_label = QLabel("")
        self.error_label.setObjectName("errorLabel")
        self.error_label.setWordWrap(True)
        command_layout.addWidget(section)
        command_layout.addWidget(self.command_input)
        command_layout.addWidget(self.intent_label)
        command_layout.addWidget(self.error_label)

        preset_grid = QGridLayout()
        preset_grid.setSpacing(6)
        presets = (
            ("PICK A", "Pick up the red cube", "A"),
            ("PICK B", "Pick up the blue ball", "B"),
            ("PICK C", "Pick up the green cylinder", "C"),
            ("PUSH A", "Push away the red cube", "A"),
            ("PUSH B", "Push away the blue ball", "B"),
            ("PUSH C", "Push away the green cylinder", "C"),
        )
        for index, (label, instruction, target) in enumerate(presets):
            button = QToolButton()
            button.setText(label)
            button.setProperty("target", target)
            button.clicked.connect(partial(self.command_input.setText, instruction))
            preset_grid.addWidget(button, index // 3, index % 3)
        command_layout.addLayout(preset_grid)
        command_layout.addSpacing(6)

        controls = QHBoxLayout()
        controls.setSpacing(7)
        style = QApplication.style()
        self.start_button = QPushButton("START")
        self.start_button.setObjectName("startButton")
        self.start_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.stop_button = QPushButton("STOP")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.reset_button = QPushButton("RESET")
        self.reset_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.nudge_button = QPushButton("NUDGE 4 CM")
        self.nudge_button.setObjectName("nudgeButton")
        self.nudge_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_ArrowRight))
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.reset_button)
        controls.addWidget(self.nudge_button)
        command_layout.addLayout(controls)
        side.addWidget(command_panel, 5)

        telemetry = QFrame()
        telemetry.setObjectName("panel")
        telemetry_layout = QGridLayout(telemetry)
        telemetry_layout.setContentsMargins(13, 10, 13, 10)
        telemetry_layout.setHorizontalSpacing(16)
        telemetry_layout.setVerticalSpacing(5)
        self.metrics = {}
        metric_names = (
            ("phase", "PHASE"),
            ("target", "TARGET"),
            ("gripper", "GRIPPER"),
            ("step", "STEP"),
            ("ground", "GROUNDING"),
            ("act", "ACT BUFFER"),
            ("safety", "SAFETY"),
            ("result", "LAST RUN"),
        )
        for index, (key, title) in enumerate(metric_names):
            row, column = divmod(index, 2)
            cell = QVBoxLayout()
            cell.setSpacing(1)
            label = QLabel(title)
            label.setObjectName("metricLabel")
            value = QLabel("--")
            value.setObjectName("valueLabel")
            value.setWordWrap(True)
            cell.addWidget(label)
            cell.addWidget(value)
            telemetry_layout.addLayout(cell, row, column)
            self.metrics[key] = value
        side.addWidget(telemetry, 3)
        body.addLayout(side, 5)
        root.addLayout(body, 1)

        footer = QFrame()
        footer.setObjectName("footer")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(12, 6, 12, 6)
        self.scene_label = QLabel("SCENE // NOT READY")
        self.scene_label.setObjectName("appSubtitle")
        self.copyright_label = QLabel("COPYRIGHT 2026 @ SUPERJACK")
        self.copyright_label.setObjectName("appSubtitle")
        self.copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.footer_message = QLabel("LOADING POLICY AND MUJOCO RUNTIME")
        self.footer_message.setObjectName("appSubtitle")
        footer_layout.addWidget(self.scene_label)
        footer_layout.addStretch(1)
        footer_layout.addWidget(self.copyright_label)
        footer_layout.addStretch(1)
        footer_layout.addWidget(self.footer_message)
        root.addWidget(footer)
        self._update_controls()

    def _connect_ui(self) -> None:
        self.command_input.textChanged.connect(self._resolve_input)
        self.start_button.clicked.connect(self._start_episode)
        self.stop_button.clicked.connect(self._stop_episode)
        self.reset_button.clicked.connect(self._reset_scene)
        self.nudge_button.clicked.connect(self._request_nudge)
        self.front_camera.clicked.connect(self._switch_main_camera)
        self.agent_camera.clicked.connect(self._switch_main_camera)
        self.wrist_camera.clicked.connect(self._switch_main_camera)

    def _start_backend(
        self,
        policy_path: str,
        seed: int,
        max_steps: int,
        local_files_only: bool,
    ) -> None:
        self.backend = DemoProcessController(
            policy_path,
            seed,
            max_steps,
            local_files_only,
        )
        self.backend.initialized.connect(self._on_initialized)
        self.backend.scene_ready.connect(self._on_scene_ready)
        self.backend.episode_ready.connect(self._on_episode_ready)
        self.backend.step_ready.connect(self._on_step)
        self.backend.episode_finished.connect(self._on_episode_finished)
        self.backend.nudge_status.connect(self._on_nudge_status)
        self.backend.error.connect(self._on_error)
        self.backend.shutdown_complete.connect(self._on_shutdown_complete)
        self.backend.start()

    def _set_status(self, text: str, state: str = "ready") -> None:
        self.status_badge.setText(text)
        self.status_badge.setProperty("state", state)
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)

    @Slot(str)
    def _resolve_input(self, text: str) -> None:
        try:
            self.resolved_instruction = self.resolver.resolve(text)
        except InstructionResolutionError as error:
            self.resolved_instruction = None
            self.intent_label.setText("COMMAND NOT READY")
            self.error_label.setText(error.message)
            self.error_label.setProperty("state", "error")
        else:
            resolved = self.resolved_instruction
            self.intent_label.setText(f"UNDERSTOOD // {resolved.display_name}")
            self.error_label.setText(
                f"POLICY PROMPT // {resolved.model_instruction}"
            )
            self.error_label.setProperty("state", "valid")
        self.error_label.style().unpolish(self.error_label)
        self.error_label.style().polish(self.error_label)
        self._update_controls()

    @Slot(object)
    def _on_initialized(self, payload: dict) -> None:
        self._ready = True
        self._set_status("SYSTEM READY")
        self.footer_message.setText(
            f"DEVICE // {payload['device'].upper()} // CONTROL // 20 HZ"
        )
        self._update_controls()
        if self._smoke_command:
            self.command_input.setText(self._smoke_command)
            QTimer.singleShot(250, self._start_episode)
        elif self._close_when_ready:
            QTimer.singleShot(400, self.close)

    @Slot(object)
    def _on_scene_ready(self, payload: dict) -> None:
        self._video_frozen = False
        self._set_frames(payload, force=True)
        self._scene_consumed = False
        self.front_camera.clear_result()
        self.scene_label.setText(
            f"SCENE // {payload['scene_seed']} // REJECTIONS // "
            f"{payload['scene_rejections']}"
        )
        if not self._running:
            self._set_status("SYSTEM READY")
            self.footer_message.setText("SCENE RANDOMIZED // AWAITING COMMAND")
        self._clear_metrics()
        self._update_controls()

    @Slot(object)
    def _on_episode_ready(self, payload: dict) -> None:
        self._nudge_available = bool(payload["nudge_available"])
        self._set_status("POLICY RUNNING", "busy")
        self.footer_message.setText(
            f"MODEL PROMPT // {payload['model_instruction']}"
        )
        self.metrics["result"].setText("RUNNING")
        self._update_controls()
        if self._smoke_nudge:
            QTimer.singleShot(150, self._request_nudge)

    @Slot(object)
    def _on_step(self, payload: dict) -> None:
        self._set_frames(payload)
        phase = PHASE_NAMES.get(payload["executed_phase"], "unknown")
        self.metrics["phase"].setText(phase.replace("_", " ").upper())
        self.metrics["target"].setText(payload["predicted_target_id"])
        self.metrics["gripper"].setText(
            "CLOSED" if payload["gripper_command"] > 0 else "OPEN"
        )
        self.metrics["step"].setText(
            f"{payload['step']} / {payload['max_steps']}"
        )
        self.metrics["ground"].setText(
            f"{payload['grounding_error_cm']:.2f} CM"
        )
        self.metrics["act"].setText(
            f"{payload['temporal_contributors']} PRED // "
            f"AGE {payload['oldest_prediction_age']}"
        )
        self.metrics["safety"].setText(
            str(payload["safety_interventions"])
        )
        if payload["perturbation_event_count"]:
            self.footer_message.setText("TARGET NUDGE APPLIED // REACQUIRING")
            self._nudge_available = False
            self._update_controls()

    @Slot(object)
    def _on_episode_finished(self, row: dict) -> None:
        self._running = False
        self._scene_consumed = True
        self._nudge_available = False
        self._video_frozen = True
        termination = row["termination"]
        if termination == "user_stop":
            self.front_camera.show_result("STOPPED", "stop")
            self._set_status("RUN STOPPED", "busy")
            self.metrics["result"].setText("STOPPED")
        elif row["task_success"]:
            self.front_camera.show_result(
                "SUCCESS",
                "success",
                animate=True,
            )
            self._set_status("TASK SUCCESS")
            self.metrics["result"].setText("SUCCESS")
        else:
            self.front_camera.show_result("FAILED", "fail")
            self._set_status("TASK FAILED", "error")
            self.metrics["result"].setText(
                row["failure_category"].replace("_", " ").upper()
            )
        self.footer_message.setText(
            f"STEPS // {row['steps']} // TERMINATION // {termination.upper()} // "
            "RESET TO RUN AGAIN"
        )
        self._update_controls()
        if self._smoke_command:
            print(
                "DEMO_SMOKE_RESULT "
                f"success={row['task_success']} "
                f"steps={row['steps']} "
                f"termination={row['termination']} "
                f"nudge={row.get('actual_delta_norm_m', 0.0):.3f}",
                flush=True,
            )
            QTimer.singleShot(300, self.close)
            return
        if self._reset_after_run:
            self._reset_after_run = False
            self._begin_reset()

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._running = False
        self._nudge_available = False
        self._video_frozen = True
        self._set_status("SYSTEM ERROR", "error")
        self.error_label.setText(message)
        self.front_camera.show_result("ERROR", "fail")
        self.footer_message.setText("RESET THE SCENE OR RESTART THE APP")
        self._update_controls()

    def _set_frames(self, payload: dict, force: bool = False) -> None:
        if self._video_frozen and not force:
            return
        for source_key in CAMERA_SOURCES:
            frame = payload.get(source_key)
            if frame is not None:
                self._latest_frames[source_key] = frame
        self._render_camera_slots()

    def _render_camera_slots(self) -> None:
        camera_slots = {
            "main": self.front_camera,
            "aux_left": self.agent_camera,
            "aux_right": self.wrist_camera,
        }
        for slot_id, camera in camera_slots.items():
            source_key = self._camera_slot_sources[slot_id]
            camera.set_source(source_key)
            camera.set_frame(self._latest_frames.get(source_key))

    @Slot(str)
    def _switch_main_camera(self, slot_id: str) -> None:
        if slot_id == "main" or slot_id not in self._camera_slot_sources:
            return
        self._camera_slot_sources["main"], self._camera_slot_sources[slot_id] = (
            self._camera_slot_sources[slot_id],
            self._camera_slot_sources["main"],
        )
        self._render_camera_slots()
        main_title = CAMERA_SOURCES[self._camera_slot_sources["main"]][0]
        self.footer_message.setText(
            f"MAIN MONITOR // {main_title.replace(' // ', ' ')}"
        )

    def _start_episode(self) -> None:
        if self.resolved_instruction is None or not self._ready or self._running:
            return
        self._running = True
        self._scene_consumed = True
        self._nudge_available = False
        self._video_frozen = False
        self.front_camera.clear_result()
        self.command_input.setEnabled(False)
        self._set_status("PREPARING RUN", "busy")
        self.footer_message.setText("VALIDATING SCENE AND STARTING POLICY")
        self._update_controls()
        if self.backend is not None:
            self.backend.start_task(self.resolved_instruction)

    def _stop_episode(self) -> None:
        if self._running and self.backend is not None:
            self.backend.request_stop()
            self._set_status("STOP REQUESTED", "busy")
            self.footer_message.setText("STOPPING AFTER CURRENT CONTROL STEP")

    def _reset_scene(self) -> None:
        if not self._ready:
            return
        if self._running:
            self._reset_after_run = True
            self._stop_episode()
            self.footer_message.setText("RESET QUEUED")
            return
        self._begin_reset()

    def _begin_reset(self) -> None:
        self._video_frozen = False
        self.front_camera.clear_result()
        self._set_status("RANDOMIZING SCENE", "busy")
        self.footer_message.setText("GENERATING COLLISION-SAFE OBJECT POSITIONS")
        self.command_input.setEnabled(False)
        self.reset_button.setEnabled(False)
        if self.backend is not None:
            self.backend.reset_scene()

    def _request_nudge(self) -> None:
        if not self._running or self.backend is None:
            return
        if self.backend.request_nudge():
            self._nudge_available = False
            self.nudge_button.setEnabled(False)
            self.footer_message.setText(
                "4 CM NUDGE QUEUED // WAITING FOR SAFE REACTIVE PHASE"
            )
        else:
            self.error_label.setText(
                "The current scene or phase cannot accept another target nudge."
            )

    @Slot(object)
    def _on_nudge_status(self, payload: dict) -> None:
        if payload.get("accepted"):
            self.footer_message.setText(
                "4 CM NUDGE QUEUED // WAITING FOR SAFE REACTIVE PHASE"
            )
            return
        self.error_label.setText(
            "The current scene cannot accept a collision-safe target nudge."
        )
        self.error_label.setProperty("state", "error")
        self.error_label.style().unpolish(self.error_label)
        self.error_label.style().polish(self.error_label)

    def _clear_metrics(self) -> None:
        for value in self.metrics.values():
            value.setText("--")

    def _update_controls(self) -> None:
        can_start = bool(
            self._ready
            and not self._running
            and not self._scene_consumed
            and self.resolved_instruction is not None
        )
        self.start_button.setEnabled(can_start)
        self.stop_button.setEnabled(self._running)
        self.reset_button.setEnabled(self._ready and not self._closing)
        self.nudge_button.setEnabled(
            self._running and self._nudge_available
        )
        self.command_input.setEnabled(
            self._ready and not self._running and not self._closing
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if self._allow_close or self.backend is None:
            event.accept()
            return
        event.ignore()
        if self._closing:
            return
        self._closing = True
        self._set_status("SHUTTING DOWN", "busy")
        self._update_controls()
        self.backend.shutdown()

    @Slot()
    def _on_shutdown_complete(self) -> None:
        if self._closing:
            self._allow_close = True
            self.close()
            return
        self.backend = None
        self._ready = False
        self._running = False
        self._set_status("RUNTIME OFFLINE", "error")
        self.footer_message.setText("RESTART THE APP TO RELOAD THE RUNTIME")
        self._update_controls()

    def populate_preview(self, frames: dict[str, np.ndarray]) -> None:
        """Populate the window without starting MuJoCo for visual QA."""
        self._set_frames(frames)
        self._ready = True
        self._set_status("SYSTEM READY")
        self.scene_label.setText("SCENE // PREVIEW // REJECTIONS // 0")
        self.footer_message.setText("DEVICE // MPS // CONTROL // 20 HZ")
        self._update_controls()
