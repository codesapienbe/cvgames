import sys
import json
import re
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, 
    QListWidget, QTextEdit, QHBoxLayout, QSplitter, QFrame
)
from PyQt6.QtCore import QThread, pyqtSignal
import subprocess

# Load games from JSON
with open('games.json', 'r', encoding='utf-8') as f:
    games_data = json.load(f)
games = games_data['games']

def title_to_module(title):
    return re.sub(r'[^a-zA-Z0-9]', '', title.strip().lower().replace(' ', ''))

class Worker(QThread):
    command_executed = pyqtSignal(str)
    output_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, command):
        super().__init__()
        self.command = command

    def run(self):
        try:
            self.command_executed.emit(self.command)
            process = subprocess.Popen(
                self.command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )
            while True:
                output = process.stdout.readline()
                error = process.stderr.readline()
                if output == '' and error == '' and process.poll() is not None:
                    break
                if output:
                    self.output_received.emit(output.strip())
                if error:
                    self.error_occurred.emit(error.strip())
            self.finished.emit(True)
        except Exception as e:
            self.error_occurred.emit(str(e))
            self.finished.emit(False)

class GameStore(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV Game Store")
        self.setGeometry(100, 100, 900, 600)

        self.worker = None
        self.init_ui()

    def init_ui(self):
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Splitter for main content
        splitter = QSplitter()
        layout.addWidget(splitter)

        # Left side: Game List
        self.game_list = QListWidget()
        self.game_list.itemClicked.connect(self.on_game_clicked)
        self.populate_games()
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.Shape.StyledPanel)
        left_layout = QVBoxLayout(left_frame)
        left_layout.addWidget(QLabel("Games"))
        left_layout.addWidget(self.game_list)
        splitter.addWidget(left_frame)

        # Right side: Game Details + Console
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.game_details = QLabel("Click a game for details.")
        self.game_details.setWordWrap(True)
        self.launch_btn = QPushButton("Launch Game")
        self.launch_btn.clicked.connect(self.launch_current_game)
        self.launch_btn.setEnabled(False)
        right_layout.addWidget(self.game_details)
        right_layout.addWidget(self.launch_btn)

        # Console output area
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        console_label = QLabel("Console Output")
        right_layout.addWidget(console_label)
        right_layout.addWidget(self.console)
        splitter.addWidget(right_widget)

    def populate_games(self):
        self.game_list.clear()
        for game in games:
            self.game_list.addItem(f"{game['title']} ({game['category']})")

    def on_game_clicked(self, item):
        idx = self.game_list.row(item)
        game = games[idx]
        details = f"<h3>{game['title']}</h3><p>{game['description']}</p><p><b>Category</b>: {game['category']}<br><b>Age Range</b>: {game['age_range']}<br><b>Difficulty</b>: {game['difficulty']}<br><b>Duration</b>: {game['duration']}</p>"
        self.game_details.setText(details)
        self.launch_btn.setEnabled(True)
        self.current_game = games[idx]

    def launch_current_game(self):
        if self.worker and self.worker.isRunning():
            self.console.append("Please wait, a game is already running!")
            return
        title = self.current_game["title"]
        module = title_to_module(title)
        command = f"uv run {module}"
        self.worker = Worker(command)
        self.worker.command_executed.connect(
            lambda cmd: self.console.append(f"Executing: {cmd}\n"))
        self.worker.output_received.connect(
            lambda out: self.console.append(f"Output: {out}"))
        self.worker.error_occurred.connect(
            lambda err: self.console.append(f"Error: {err}"))
        self.worker.finished.connect(
            lambda: self.console.append(f"Finished: {title}"))
        self.worker.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GameStore()
    window.show()
    sys.exit(app.exec())

