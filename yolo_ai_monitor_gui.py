import base64
import csv
from datetime import datetime, timedelta
import importlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import cv2
import customtkinter as ctk
import numpy as np
import psutil
import requests
from ultralytics import YOLO

# ==========================================
# 1. UIテーマおよびグローバル定数の設定
# ==========================================
ctk.set_appearance_mode("Dark")  # ダークモードを適用
ctk.set_default_color_theme("blue")  # テーマカラーをブルーに設定

# COCO データセットの 80 クラス（ダイアログ表示用にカテゴリ分け）
COCO_CLASSES_BY_CATEGORY = {
    "人物・人": ["person"],
    "乗り物・交通": [
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
    ],
    "屋外・動物": [
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ],
    "持ち物・衣類": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    "身の回り品": [
        "bottle",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "tv",
        "laptop",
        "cell phone",
    ],
}

# 動作環境チェック用：必須ライブラリと PyPI パッケージ名のマッピング
REQUIRED_PACKAGES = {
    "customtkinter": "customtkinter",
    "cv2": "opencv-python",
    "ultralytics": "ultralytics",
    "openvino": "openvino",
    "psutil": "psutil",
    "requests": "requests",
    "torch": "torch",
}


# ==========================================
# 2. 検出対象クラス選択 サブウィンドウ (モーダル)
# ==========================================
class ClassSelectionWindow(ctk.CTkToplevel):

  def __init__(self, parent, selected_classes_set, on_apply_callback):
    """検出対象オブジェクトをカテゴリ一覧からチェックボックスで一括設定するモーダルダイアログ"""
    super().__init__(parent)

    self.title("検出対象クラス 一覧設定")
    self.geometry("680x550")
    self.grab_set()  # 親ウィンドウへの操作をブロック（モーダル表示）

    self.selected_set = set(selected_classes_set)
    self.on_apply_callback = on_apply_callback
    self.checkboxes = {}

    self._create_widgets()

  def _create_widgets(self):
    """サブウィンドウ内UIコンポーネントの構築"""
    main_frame = ctk.CTkFrame(self)
    main_frame.pack(fill="both", expand=True, padx=15, pady=15)

    # 全選択 / 全解除ボタンバー
    top_bar = ctk.CTkFrame(main_frame, fg_color="transparent")
    top_bar.pack(fill="x", pady=5)

    ctk.CTkButton(
        top_bar,
        text="すべて選択",
        width=100,
        command=self._select_all,
        fg_color="gray",
    ).pack(side="left", padx=5)
    ctk.CTkButton(
        top_bar,
        text="すべて解除",
        width=100,
        command=self._deselect_all,
        fg_color="gray",
    ).pack(side="left", padx=5)

    # カテゴリ別スクロール表示エリア
    scroll_frame = ctk.CTkScrollableFrame(main_frame)
    scroll_frame.pack(fill="both", expand=True, pady=10)

    for category, cls_list in COCO_CLASSES_BY_CATEGORY.items():
      # カテゴリ見出し
      cat_label = ctk.CTkLabel(
          scroll_frame, text=f"■ {category}", font=("メイリオ", 12, "bold")
      )
      cat_label.pack(anchor="w", pady=(10, 2))

      # 3列グリッドでチェックボックスを配置
      grid_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
      grid_frame.pack(fill="x", padx=10, pady=2)

      for i, cls_name in enumerate(cls_list):
        var = ctk.BooleanVar(value=(cls_name in self.selected_set))
        chk = ctk.CTkCheckBox(grid_frame, text=cls_name, variable=var)
        chk.grid(row=i // 3, column=i % 3, sticky="w", padx=10, pady=4)
        self.checkboxes[cls_name] = var

    # 下部 確定ボタン
    btn_apply = ctk.CTkButton(
        main_frame,
        text="設定を適用して閉じる",
        fg_color="green",
        hover_color="darkgreen",
        command=self._apply_and_close,
    )
    btn_apply.pack(fill="x", pady=5)

  def _select_all(self):
    """すべてのチェックボックスをオン"""
    for var in self.checkboxes.values():
      var.set(True)

  def _deselect_all(self):
    """すべてのチェックボックスをオフ"""
    for var in self.checkboxes.values():
      var.set(False)

  def _apply_and_close(self):
    """選択状況を親ウィンドウに通知してダイアログを破棄"""
    res = [cls for cls, var in self.checkboxes.items() if var.get()]
    self.on_apply_callback(res)
    self.destroy()


# ==========================================
# 3. メインアプリケーション クラス
# ==========================================
class SurveillanceAIMonitor(ctk.CTk):

  def __init__(self):
    """メインウィンドウの初期化および初期状態のセットアップ"""
    super().__init__()

    self.title("防犯カメラ AI 異常検知システム Pro (YOLO + LM Studio)")
    self.geometry("1020x960")

    # 動作フラグおよびスレッド管理変数
    self.is_running = False
    self.stop_requested = False
    self.analysis_thread = None

    # デフォルトの検出対象クラス (人物、車、バイク、自転車)
    self.selected_target_classes = ["person", "car", "motorcycle", "bicycle"]
    # デバッグログファイルの保存パス
    self.debug_log_path = os.path.join(
        os.getcwd(), "surveillance_ai_debug.log"
    )

    # 実行環境のハードウェアアクセラレータ自動検出
    self.available_devices = self._detect_hardware_devices()

    # GUIの構築とリソース監視スレッドの起動
    self._create_widgets()
    self._start_resource_monitoring()

  # ------------------------------------------
  # ハードウェア検出ロジック
  # ------------------------------------------
  def _detect_hardware_devices(self):
    """実行PCのハードウェア構成 (CPU, Intel iGPU / NPU, NVIDIA dGPU) を自動判定"""
    devices = {"CPU": "CPU (汎用)"}

    # 1. OpenVINO デバイススキャン (Intel CPU / iGPU / NPU)
    try:
      import openvino as ov

      core = ov.Core()
      ov_devs = [str(d).upper() for d in core.available_devices]

      if any("GPU" in d for d in ov_devs):
        devices["iGPU"] = "Intel iGPU (OpenVINO)"
      else:
        devices["iGPU"] = "Intel iGPU (OpenVINO対応)"

      if any("NPU" in d for d in ov_devs):
        devices["NPU"] = "Intel NPU (OpenVINO)"
      else:
        devices["NPU"] = "Intel NPU (OpenVINO対応)"
    except Exception as e:
      self._write_debug_file(f"[Hardware Detect] OpenVINO init error: {e}")
      devices["iGPU"] = "Intel iGPU (OpenVINO)"
      devices["NPU"] = "Intel NPU (OpenVINO)"

    # 2. NVIDIA dGPU 検出 (PyTorch または nvidia-smi 経由の二重判定)
    has_nvidia = False
    gpu_name_str = "NVIDIA dGPU"

    # 方法A: PyTorch / CUDA による判定
    try:
      import torch

      if torch.cuda.is_available():
        gpu_name_str = f"NVIDIA dGPU ({torch.cuda.get_device_name(0)})"
        has_nvidia = True
    except Exception:
      pass

    # 方法B: nvidia-smi コマンドの直接実行による判定
    if not has_nvidia:
      try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0 and res.stdout.strip():
          gpu_name_str = f"NVIDIA dGPU ({res.stdout.strip().splitlines()[0]})"
          has_nvidia = True
      except Exception:
        pass

    if has_nvidia:
      devices["dGPU"] = gpu_name_str
    else:
      devices["dGPU"] = "NVIDIA dGPU (未検出/CUDA未設定)"

    return devices

  # ------------------------------------------
  # GUI コンポーネント構築
  # ------------------------------------------
  def _create_widgets(self):
    """CustomTkinter を使用したGUIレイアウトの初期化"""
    main_frame = ctk.CTkFrame(self)
    main_frame.pack(fill="both", expand=True, padx=15, pady=15)

    # 1. 動画フォルダ / 保存先 設置セクション
    folder_frame = ctk.CTkFrame(main_frame)
    folder_frame.pack(fill="x", padx=10, pady=5)

    ctk.CTkLabel(folder_frame, text="動画フォルダ:").grid(
        row=0, column=0, padx=5, pady=5, sticky="w"
    )
    self.entry_video_dir = ctk.CTkEntry(folder_frame, width=540)
    self.entry_video_dir.insert(
        0, r"未指定"
    )
    self.entry_video_dir.grid(row=0, column=1, padx=5, pady=5)
    ctk.CTkButton(
        folder_frame, text="参照", width=70, command=self._browse_video_dir
    ).grid(row=0, column=2, padx=5, pady=5)

    ctk.CTkLabel(folder_frame, text="保存フォルダ:").grid(
        row=1, column=0, padx=5, pady=5, sticky="w"
    )
    self.entry_save_dir = ctk.CTkEntry(folder_frame, width=540)
    self.entry_save_dir.insert(
        0, r"未指定"
    )
    self.entry_save_dir.grid(row=1, column=1, padx=5, pady=5)
    ctk.CTkButton(
        folder_frame, text="参照", width=70, command=self._browse_save_dir
    ).grid(row=1, column=2, padx=5, pady=5)

    # 2. 推論ハードウェア & YOLOモデル設定セクション
    hw_frame = ctk.CTkFrame(main_frame)
    hw_frame.pack(fill="x", padx=10, pady=5)

    ctk.CTkLabel(hw_frame, text="推論デバイス:").grid(
        row=0, column=0, padx=5, pady=5, sticky="w"
    )
    dev_options = list(self.available_devices.values())
    self.combo_device = ctk.CTkOptionMenu(
        hw_frame, values=dev_options, width=240
    )
    # デフォルトで Intel iGPU、なければ CPU を初期選択
    default_dev = self.available_devices.get(
        "iGPU", self.available_devices["CPU"]
    )
    self.combo_device.set(default_dev)
    self.combo_device.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    ctk.CTkLabel(hw_frame, text="YOLOモデル:").grid(
        row=0, column=2, padx=5, pady=5, sticky="w"
    )
    self.combo_yolo_model = ctk.CTkOptionMenu(
        hw_frame,
        values=[
            "yolo26s_int8_openvino_model",
            "yolo26s.pt",
            "yolo26n.pt",
            "yolo11s.pt",
            "yolov8s.pt",
        ],
        width=230,
    )
    self.combo_yolo_model.grid(row=0, column=3, padx=5, pady=5, sticky="w")

    ctk.CTkButton(
        hw_frame,
        text="🤖 AI自動最適設定",
        width=140,
        fg_color="#8B5CF6",
        hover_color="#7C3AED",
        command=self._auto_config_by_ai,
    ).grid(row=0, column=4, padx=10, pady=5, sticky="e")

    # 3. 検出対象クラス ＆ デバッグ設定セクション
    class_frame = ctk.CTkFrame(main_frame)
    class_frame.pack(fill="x", padx=10, pady=5)

    ctk.CTkLabel(
        class_frame, text="検出対象クラス:", font=("メイリオ", 11, "bold")
    ).grid(row=0, column=0, padx=5, pady=5, sticky="w")

    ctk.CTkButton(
        class_frame,
        text="🎯 検出クラスを一覧で設定...",
        width=200,
        fg_color="#0284C7",
        hover_color="#0369A1",
        command=self._open_class_selection_dialog,
    ).grid(row=0, column=1, padx=10, pady=5, sticky="w")

    self.lbl_selected_classes_summary = ctk.CTkLabel(
        class_frame,
        text="選択中: person, car, motorcycle, bicycle",
        text_color="lightgray",
    )
    self.lbl_selected_classes_summary.grid(
        row=0, column=2, padx=10, pady=5, sticky="w"
    )

    self.chk_debug_mode_var = ctk.BooleanVar(value=False)
    ctk.CTkSwitch(
        class_frame,
        text="🐞 デバッグモード (詳細ログをファイル出力)",
        variable=self.chk_debug_mode_var,
    ).grid(row=0, column=3, padx=15, pady=5, sticky="e")

    # 4. フレームサンプリング間隔 & 動的追跡設定
    sample_frame = ctk.CTkFrame(main_frame)
    sample_frame.pack(fill="x", padx=10, pady=5)

    ctk.CTkLabel(sample_frame, text="通常時サンプリング(秒):").grid(
        row=0, column=0, padx=5, pady=5, sticky="w"
    )
    self.entry_normal_sample = ctk.CTkEntry(sample_frame, width=70)
    self.entry_normal_sample.insert(0, "0.5")
    self.entry_normal_sample.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    ctk.CTkLabel(
        sample_frame, text="検知時サンプリング(秒 / 0で全フレーム):"
    ).grid(row=0, column=2, padx=15, pady=5, sticky="w")
    self.entry_high_sample = ctk.CTkEntry(sample_frame, width=70)
    self.entry_high_sample.insert(0, "0.1")
    self.entry_high_sample.grid(row=0, column=3, padx=5, pady=5, sticky="w")

    ctk.CTkLabel(sample_frame, text="検知時追跡時間(秒):").grid(
        row=0, column=4, padx=15, pady=5, sticky="w"
    )
    self.entry_tracking_dur = ctk.CTkEntry(sample_frame, width=70)
    self.entry_tracking_dur.insert(0, "5.0")
    self.entry_tracking_dur.grid(row=0, column=5, padx=5, pady=5, sticky="w")

    # 5. LM Studio 接続 & VLMモデル設定
    lm_frame = ctk.CTkFrame(main_frame)
    lm_frame.pack(fill="x", padx=10, pady=5)

    ctk.CTkLabel(lm_frame, text="LM Studio URL:").grid(
        row=0, column=0, padx=5, pady=5, sticky="w"
    )
    self.entry_lm_url = ctk.CTkEntry(lm_frame, width=300)
    self.entry_lm_url.insert(0, "http://localhost:1234/v1/chat/completions")
    self.entry_lm_url.grid(
        row=0, column=1, columnspan=2, padx=5, pady=5, sticky="w"
    )

    ctk.CTkButton(
        lm_frame,
        text="接続テスト",
        width=85,
        fg_color="gray",
        command=self._test_lm_connection,
    ).grid(row=0, column=3, padx=5, pady=5, sticky="w")

    ctk.CTkLabel(lm_frame, text="VLMモデル名:").grid(
        row=1, column=0, padx=5, pady=5, sticky="w"
    )
    self.entry_lm_model = ctk.CTkEntry(lm_frame, width=300)
    self.entry_lm_model.insert(0, "google/gemma-4-e4b")
    self.entry_lm_model.grid(
        row=1, column=1, columnspan=2, padx=5, pady=5, sticky="w"
    )

    ctk.CTkButton(
        lm_frame,
        text="モデル自動取得",
        width=110,
        fg_color="#10B981",
        command=self._fetch_lm_models,
    ).grid(row=1, column=3, padx=5, pady=5, sticky="w")

    ctk.CTkLabel(lm_frame, text="Max Tokens:").grid(
        row=0, column=4, padx=10, pady=5, sticky="w"
    )
    self.entry_max_tokens = ctk.CTkEntry(lm_frame, width=70)
    self.entry_max_tokens.insert(0, "4000")
    self.entry_max_tokens.grid(row=0, column=5, padx=5, pady=5, sticky="w")

    ctk.CTkLabel(lm_frame, text="デコードスレッド数:").grid(
        row=1, column=4, padx=10, pady=5, sticky="w"
    )
    self.entry_threads = ctk.CTkEntry(lm_frame, width=70)
    self.entry_threads.insert(0, "4")
    self.entry_threads.grid(row=1, column=5, padx=5, pady=5, sticky="w")

    # 6. AIプロンプトテキストボックス
    prompt_frame = ctk.CTkFrame(main_frame)
    prompt_frame.pack(fill="x", padx=10, pady=5)

    ctk.CTkLabel(prompt_frame, text="AI解析プロンプト:").pack(
        anchor="w", padx=5, pady=2
    )
    self.text_prompt = ctk.CTkTextbox(prompt_frame, height=80)
    self.text_prompt.insert(
        "1.0",
        "定点防犯カメラ画像の時系列解析を行ってください。\n"
        "【手順】\n"
        "1. 人物・車両・動物等の動きや滞留を時系列で確認。\n"
        "2. 侵入・不審行為・長時間の滞留等の「異常」があるか判定。\n\n"
        "※必ず以下の形式で簡潔に出力してください（概要は60文字以内）：\n"
        "【判定】: 異常あり (または 異常なし)\n"
        "【異常フレーム】: 1, 5 (該当する画像の番号を半角数字カンマ区切りで列挙)\n"
        "【概要】: (簡潔な理由)",
    )
    self.text_prompt.pack(fill="x", padx=5, pady=5)

    # 7. 制御ボタン & リアルタイム進捗ステータス表示
    ctrl_frame = ctk.CTkFrame(main_frame)
    ctrl_frame.pack(fill="x", padx=10, pady=10)

    self.btn_start = ctk.CTkButton(
        ctrl_frame,
        text="解析開始",
        fg_color="green",
        hover_color="darkgreen",
        command=self._start_analysis,
    )
    self.btn_start.pack(side="left", padx=10, pady=5)

    self.btn_stop = ctk.CTkButton(
        ctrl_frame,
        text="停止",
        fg_color="red",
        hover_color="darkred",
        state="disabled",
        command=self._stop_analysis,
    )
    self.btn_stop.pack(side="left", padx=10, pady=5)

    # プログレスバー ＆ 詳細進捗テキスト（処理中ファイル名・%）
    progress_container = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
    progress_container.pack(
        side="left", fill="x", expand=True, padx=15, pady=2
    )

    self.lbl_progress_info = ctk.CTkLabel(
        progress_container,
        text="待機中...",
        anchor="w",
        font=("メイリオ", 11, "bold"),
    )
    self.lbl_progress_info.pack(fill="x", pady=2)

    self.progress_bar = ctk.CTkProgressBar(progress_container)
    self.progress_bar.set(0)
    self.progress_bar.pack(fill="x", pady=2)

    # 8. コンソールログ出力枠
    log_frame = ctk.CTkFrame(main_frame)
    log_frame.pack(fill="both", expand=True, padx=10, pady=5)

    self.text_log = ctk.CTkTextbox(log_frame, font=("Consolas", 11))
    self.text_log.pack(fill="both", expand=True, padx=5, pady=5)

    # 9. 最下部 CPU/RAM ステータスバー
    self.status_bar = ctk.CTkLabel(
        self,
        text="CPU: --% | RAM: --%",
        anchor="w",
        font=("Consolas", 11),
    )
    self.status_bar.pack(fill="x", side="bottom", padx=15, pady=3)

  # ------------------------------------------
  # イベントハンドラ ＆ ダイアログ処理
  # ------------------------------------------
  def _open_class_selection_dialog(self):
    """クラス選択モーダルを開き、確定時に概要テキストを更新"""

    def on_apply(new_classes):
      self.selected_target_classes = new_classes
      summary_str = ", ".join(new_classes[:4])
      if len(new_classes) > 4:
        summary_str += f" (+{len(new_classes)-4}個)"
      self.lbl_selected_classes_summary.configure(
          text=f"選択中: {summary_str if new_classes else '全クラス対象'}"
      )
      self.log(
          f"🎯 検出対象クラスを更新しました ({len(new_classes)}個)"
      )

    ClassSelectionWindow(self, self.selected_target_classes, on_apply)

  def _write_debug_file(self, msg):
    """デバッグモード時に外部テキストへタイムスタンプ付きで追記"""
    try:
      with open(self.debug_log_path, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
      pass

  def log(self, message):
    """GUIログエリアへの安全なメッセージ出力 (メインスレッド非同期対応)"""

    def _append():
      self.text_log.insert(
          "end", f"[{time.strftime('%H:%M:%S')}] {message}\n"
      )
      self.text_log.see("end")

    self.after(0, _append)
    if self.chk_debug_mode_var.get():
      self._write_debug_file(message)

  def _browse_video_dir(self):
    """動画フォルダ参照ダイアログ"""
    path = ctk.filedialog.askdirectory()
    if path:
      self.entry_video_dir.delete(0, "end")
      self.entry_video_dir.insert(0, path)

  def _browse_save_dir(self):
    """保存フォルダ参照ダイアログ"""
    path = ctk.filedialog.askdirectory()
    if path:
      self.entry_save_dir.delete(0, "end")
      self.entry_save_dir.insert(0, path)

  def _fetch_lm_models(self):
    """LM Studio API (/v1/models) から現在アクティブなモデルIDを自動取得"""
    url = (
        self.entry_lm_url.get().strip().replace("/chat/completions", "/models")
    )
    self.log(f"LM Studio からロード中モデルを取得中: {url}")

    def fetch():
      try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
          models_data = res.json().get("data", [])
          if models_data:
            model_id = models_data[0].get("id", "google/gemma-4-e4b")
            self.after(
                0,
                lambda: (
                    self.entry_lm_model.delete(0, "end"),
                    self.entry_lm_model.insert(0, model_id),
                ),
            )
            self.log(f"✅ LM Studio アクティブモデルをセットしました: {model_id}")
          else:
            self.log("⚠️ ロード中のモデルが見つかりませんでした。")
        else:
          self.log(f"⚠️ 取得失敗 Status: {res.status_code}")
      except Exception as e:
        self.log(f"❌ モデル取得エラー: {e}")

    threading.Thread(target=fetch, daemon=True).start()

  # ------------------------------------------
  # AI パラメーター自動最適化 エンジン
  # ------------------------------------------
  def _auto_config_by_ai(self):
    """動画スペックとPCリソースから最適な推論パラメータを自動算出"""
    video_dir = self.entry_video_dir.get().strip()
    video_files = self._get_sorted_videos(video_dir)

    cpu_cores = psutil.cpu_count(logical=True) or 4
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)

    if not video_files:
      self.log(
          "⚠️ 動画ファイルが見つからないため、ハードウェア構成のみで自動最適設定を行います。"
      )
      w, h, fps, total_f = 1920, 1080, 30.0, 900
    else:
      # 先頭のサンプル動画情報をメタデータ読み込み
      sample_video = video_files[0]
      cap = cv2.VideoCapture(sample_video)
      if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
      else:
        w, h, fps, total_f = 1920, 1080, 30.0, 900

    self.log("🤖 AIへPC構成と動画スペックを送信し、最適設定を計算中...")

    prompt = f"""あなたは動画解析の最適化AIです。以下のスペック情報から防犯カメラAI解析の最適パラメータを算出してください。

【入力パラメータ】
- 解像度: {w}x{h}
- FPS: {fps:.1f}
- CPU論理コア数: {cpu_cores}
- RAM: {ram_gb} GB
- デバイス: {self.combo_device.get()}

【要件】
応答文の中に必ず次の純粋なJSONオブジェクトのみを含めて出力してください：
{{"normal_interval": 0.5, "high_interval": 0.1, "tracking_duration": 5.0, "num_threads": 4}}
"""

    def apply_config(norm, high, dur, th):
      """GUIフォームに数値を反映"""
      self.entry_normal_sample.delete(0, "end")
      self.entry_normal_sample.insert(0, str(norm))
      self.entry_high_sample.delete(0, "end")
      self.entry_high_sample.insert(0, str(high))
      self.entry_tracking_dur.delete(0, "end")
      self.entry_tracking_dur.insert(0, str(dur))
      self.entry_threads.delete(0, "end")
      self.entry_threads.insert(0, str(th))

    def query():
      parsed_ok = False
      try:
        url = self.entry_lm_url.get().strip()
        payload = {
            "model": self.entry_lm_model.get().strip(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 300,
        }
        res = requests.post(url, json=payload, timeout=12)
        if res.status_code == 200:
          ans = res.json()["choices"][0]["message"]["content"]
          self.log(f"[AI自動設定 応答テキスト]: {ans[:150]}...")

          # 応答文から正規表現で JSON 部分のみ抽出 (思考型モデルやMarkdown装飾に対応)
          match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", ans, re.DOTALL)
          json_str = (
              match.group(1)
              if match
              else re.search(r"\{.*?\}", ans, re.DOTALL)
          )

          if json_str:
            raw_code = json_str.group(0) if hasattr(json_str, "group") else json_str
            cfg = json.loads(raw_code)
            norm = float(cfg.get("normal_interval", 0.5))
            high = float(cfg.get("high_interval", 0.1))
            dur = float(cfg.get("tracking_duration", 5.0))
            th = int(cfg.get("num_threads", max(2, cpu_cores // 2)))

            self.after(0, lambda: apply_config(norm, high, dur, th))
            self.log("✨ 【AI自動最適設定】がGUIに正常反映されました！")
            parsed_ok = True

      except Exception as e:
        self.log(f"⚠️ AI通信/解析エラー ({e})")

      # 通信エラーやJSON解釈失敗時のフォールバック（ルールベースで安全に自動判定）
      if not parsed_ok:
        self.log(
            "💡 ルールベースのハードウェア最適化ロジックで自動算出します..."
        )
        rec_threads = max(2, min(8, cpu_cores - 2 if cpu_cores > 4 else 2))
        rec_normal = 0.5 if (w * h) <= (1920 * 1080) else 1.0
        rec_high = 0.1
        rec_dur = 5.0

        self.after(
            0, lambda: apply_config(rec_normal, rec_high, rec_dur, rec_threads)
        )
        self.log(
            f"✨ 【ルールベース自動設定】（スレッド: {rec_threads}, 通常"
            f" {rec_normal}s, 検知 {rec_high}s）をセットしました！"
        )

    threading.Thread(target=query, daemon=True).start()

  def _test_lm_connection(self):
    """LM Studio への疎通テスト"""
    url = self.entry_lm_url.get().strip()
    self.log(f"LM Studio への接続テストを開始: {url}")

    def test():
      try:
        res = requests.post(
            url,
            json={
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5,
            },
            timeout=5,
        )
        if res.status_code == 200:
          self.log("✅ LM Studio 接続成功！レスポンスを確認しました。")
        else:
          self.log(f"⚠️ 接続失敗 Status: {res.status_code}")
      except Exception as e:
        self.log(f"❌ 接続エラー: {e}")

    threading.Thread(target=test, daemon=True).start()

  def _start_resource_monitoring(self):
    """1秒周期で CPU / RAM 使用率を計測してステータスバーを更新"""

    def update_resources():
      try:
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        self.status_bar.configure(text=f"CPU: {cpu}% | RAM: {ram}%")
      except Exception:
        pass
      self.after(1000, update_resources)

    update_resources()

  def _parse_video_start_datetime(self, file_path):
    """ファイルパス（例: 20260921/12/34.mp4）から実際の録画開始日時を復元"""
    match = re.search(r"(\d{8})[\\/](\d{2})[\\/](\d{2})\.mp4$", file_path)
    if match:
      date_str, hour_str, min_str = match.groups()
      try:
        dt_str = f"{date_str}{hour_str}{min_str}"
        return datetime.strptime(dt_str, "%Y%m%d%H%M")
      except ValueError:
        pass
    # フォールバック：ファイルの更新日時を使用
    try:
      return datetime.fromtimestamp(os.path.getmtime(file_path))
    except Exception:
      return datetime.now()

  def _start_analysis(self):
    """バックグラウンド解析タスクの開始"""
    self.is_running = True
    self.stop_requested = False
    self.btn_start.configure(state="disabled")
    self.btn_stop.configure(state="normal")
    self.log("=== 解析タスクを開始します ===")

    self.analysis_thread = threading.Thread(
        target=self._run_analysis_process, daemon=True
    )
    self.analysis_thread.start()

  def _stop_analysis(self):
    """停止要求の発行"""
    if self.is_running:
      self.stop_requested = True
      self.log(
          "[Info] 停止要求を送信しました... 現在の動画処理完了後に終了します。"
      )

  # ------------------------------------------
  # メイン解析 パイプライン (バックグラウンドスレッド)
  # ------------------------------------------
  def _run_analysis_process(self):
    """動画読み込み -> OpenVINO推論 -> 条件判定 -> VLMバッチ処理 のメインパイプライン"""
    video_dir = self.entry_video_dir.get().strip()
    save_dir = self.entry_save_dir.get().strip()
    model_name = self.combo_yolo_model.get().strip()
    selected_dev_text = self.combo_device.get()
    lm_url = self.entry_lm_url.get().strip()
    lm_model = self.entry_lm_model.get().strip()
    prompt_text = self.text_prompt.get("1.0", "end").strip()

    # 入力値のパースとバリデーション
    try:
      normal_interval = float(self.entry_normal_sample.get().strip())
    except ValueError:
      normal_interval = 0.5

    try:
      high_interval = float(self.entry_high_sample.get().strip())
    except ValueError:
      high_interval = 0.1

    try:
      tracking_duration = float(self.entry_tracking_dur.get().strip())
    except ValueError:
      tracking_duration = 5.0

    try:
      num_threads = int(self.entry_threads.get().strip())
    except ValueError:
      num_threads = 4

    try:
      max_tokens = int(self.entry_max_tokens.get().strip())
    except ValueError:
      max_tokens = 4000

    target_classes = self.selected_target_classes
    os.makedirs(save_dir, exist_ok=True)

    # 推論デバイス環境変数のセット
    if "iGPU" in selected_dev_text:
      os.environ["OPENVINO_DEVICE"] = "GPU"
    elif "NPU" in selected_dev_text:
      os.environ["OPENVINO_DEVICE"] = "NPU"
    else:
      os.environ["OPENVINO_DEVICE"] = "CPU"

    # OpenVINO モデルの自動量子化 / ロード
    if os.path.isdir(model_name):
      openvino_folder = model_name
    else:
      model_base = os.path.splitext(model_name)[0]
      openvino_folder = f"{model_base}_int8_openvino_model"

      if not os.path.exists(openvino_folder):
        self.log(
            f"[Info] INT8 量子化 OpenVINO モデル ({model_name}) を作成中..."
        )
        try:
          m = YOLO(model_name)
          exported_path = m.export(
              format="openvino", int8=True, imgsz=640, device="cpu"
          )
          openvino_folder = str(exported_path)
        except Exception as e:
          self.log(f"❌ モデル量子化エラー: {e}")
          self._finish_process()
          return

    self.log(
        f"[Info] OpenVINO モデル ({openvino_folder}) を [{selected_dev_text}]"
        " にロード中..."
    )
    try:
      yolo_model = YOLO(openvino_folder, task="detect")
    except Exception as e:
      self.log(f"❌ モデルロードエラー: {e}")
      self._finish_process()
      return

    # 動画ファイルの走査とソート
    video_files = self._get_sorted_videos(video_dir)
    total_videos = len(video_files)
    if total_videos == 0:
      self.log("⚠️ 対象フォルダ内に動画ファイルが見つかりませんでした。")
      self._finish_process()
      return

    self.log(
        f"[Info] 計 {total_videos} 件の動画をマルチスレッド（デコード:"
        f" {num_threads} スレッド）で走査します。"
    )

    # スレッド間通信用キュー
    frame_queue = queue.Queue(maxsize=200)
    video_queue = queue.Queue()
    for vp in video_files:
      video_queue.put(vp)

    high_freq_until = {}  # 動的追跡（高精度サンプリング）管理辞書

    # --- Worker Thread: 並列動画デコーダー ---
    def decoder_worker():
      while not self.stop_requested:
        try:
          vp = video_queue.get_nowait()
        except queue.Empty:
          break

        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
          video_queue.task_done()
          continue

        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        target_idx = 0
        while target_idx < total_f and not self.stop_requested:
          current_sec = target_idx / fps
          # 現在時刻が追跡モード期間内であれば高精度間隔を採用
          active_interval = (
              high_interval
              if current_sec < high_freq_until.get(vp, 0.0)
              else normal_interval
          )

          cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
          ret, frame = cap.read()
          if not ret:
            break

          frame_queue.put((frame, vp, current_sec))

          # 次の読み込みフレームインデックスを加算
          step = (
              1
              if active_interval <= 0.0
              else max(1, int(fps * active_interval))
          )
          target_idx += step

        cap.release()
        video_queue.task_done()

    # デコード用ワーカースレッド群の起動
    decoder_threads = []
    for _ in range(num_threads):
      t = threading.Thread(target=decoder_worker, daemon=True)
      t.start()
      decoder_threads.append(t)

    vlm_buffer = []  # VLM送信用の画像バッファ
    last_collected = {}
    prev_frames = {}
    processed_videos_set = set()

    # --- Main Loop: YOLO推論 ＆ バッチ抽出 ---
    while not self.stop_requested:
      try:
        frame, vp, sec = frame_queue.get(timeout=1.0)
      except queue.Empty:
        if video_queue.empty() and not any(
            t.is_alive() for t in decoder_threads
        ):
          break
        continue

      processed_videos_set.add(vp)
      curr_v_count = len(processed_videos_set)
      pct = (curr_v_count / total_videos) * 100
      v_name = os.path.basename(vp)

      # プログレスバーと進捗テキストの表示更新
      self.after(
          0,
          lambda c=curr_v_count, t=total_videos, p=pct, name=v_name: (
              self.lbl_progress_info.configure(
                  text=f"[{c} / {t} 件 ({p:.1f}%)] 処理中: {name}"
              ),
              self.progress_bar.set(p / 100.0),
          ),
      )

      # YOLO 物体検出推論の実行
      res = yolo_model(frame, imgsz=640, verbose=False)[0]
      impact_score, detected_info, has_person = self._calc_impact(
          res, frame.shape, yolo_model.names, target_classes
      )

      # 検出物体（人物等）があった場合の処理
      if impact_score > 0:
        # 動的サンプリング（高精度キャプチャ）タイマーを発動
        high_freq_until[vp] = sec + tracking_duration
        prev_f = prev_frames.get(vp, None)
        motion_score = self._calc_motion(prev_f, frame)
        last_s = last_collected.get(vp, -1.0)

        # 動きが確認されるか人物が含まれる場合、1秒以上の間隔でVLMバッファに追加
        if (motion_score > 0.005 or has_person) and (sec - last_s >= 1.0):
          last_collected[vp] = sec
          vlm_buffer.append({
              "frame": frame,
              "video_path": vp,
              "sec": sec,
              "has_person": has_person,
              "impact_score": impact_score,
          })

          # バッファが20枚溜まったら LM Studio へ送信
          if len(vlm_buffer) >= 20:
            self._process_vlm_batch(
                vlm_buffer[:20],
                lm_url,
                lm_model,
                save_dir,
                prompt_text,
                max_tokens,
            )
            del vlm_buffer[:20]

      prev_frames[vp] = frame
      frame_queue.task_done()

    # 残存バッファの最終解析処理
    if vlm_buffer and not self.stop_requested:
      self.log(f"[Info] 残り {len(vlm_buffer)} 枚を最終解析します。")
      self._process_vlm_batch(
          vlm_buffer, lm_url, lm_model, save_dir, prompt_text, max_tokens
      )
      vlm_buffer.clear()

    self.log(
        "=== 解析が正常終了しました ==="
        if not self.stop_requested
        else "=== 処理を途中停止しました ==="
    )
    self._finish_process()

  # ------------------------------------------
  # VLM (Vision-Language Model) 送信・保存処理
  # ------------------------------------------
  def _process_vlm_batch(
      self, batch_items, url, model_id, save_folder, prompt_text, max_tokens
  ):
    """フレーム群を Base64 にエンコードし、LM Studio へマルチモーダル解析リクエスト"""
    self.log(
        f"🚀 【AI解析】 {len(batch_items)} 枚のフレームを AI ({model_id})"
        " へ送信中..."
    )
    batch_items.sort(key=lambda x: (x["video_path"], x["sec"]))

    content_list = [{"type": "text", "text": prompt_text}]
    for item in batch_items:
      h, w = item["frame"].shape[:2]
      # 高精細画像は通信量削減のため最大800pxにリサイズ
      if max(h, w) > 800:
        scale = 800 / float(max(h, w))
        resized = cv2.resize(
            item["frame"],
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
      else:
        resized = item["frame"]

      # JPEGエンコード -> Base64変換
      _, buffer = cv2.imencode(
          ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
      )
      b64 = base64.b64encode(buffer).decode("utf-8")
      content_list.append({
          "type": "image_url",
          "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
      })

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": content_list}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }

    try:
      response = requests.post(url, json=payload, timeout=180)
      if response.status_code == 200:
        ans = response.json()["choices"][0]["message"]["content"]
        self.log(
            f"\n--- AI ({model_id}) 解析結果 ---\n{ans}\n----------------------------"
        )

        is_abnormal = "【判定】: 異常あり" in ans or "【判定】:異常あり" in ans
        saved_indices = set()

        # 異常が検出された場合の個別フレーム保存
        if is_abnormal:
          match = re.search(r"【異常フレーム】:\s*(.*)", ans)
          target_indices = []
          if match:
            frame_str = match.group(1).strip()
            if "all" in frame_str.lower():
              target_indices = list(range(1, len(batch_items) + 1))
            else:
              found_nums = re.findall(r"\d+", frame_str)
              target_indices = [
                  int(n) for n in found_nums if 1 <= int(n) <= len(batch_items)
              ]
          if not target_indices:
            target_indices = list(range(1, len(batch_items) + 1))

          for idx in target_indices:
            item = batch_items[idx - 1]
            fn = self._save_frame_with_real_time(
                item["frame"],
                item["video_path"],
                item["sec"],
                save_folder,
                prefix="ABNORMAL",
            )
            saved_indices.add(idx)
            self.log(f"  └ 💾 [異常保存] No.{idx:02d} -> {fn}")
            self._save_csv_log(
                save_folder, item["video_path"], item["sec"], "ABNORMAL", ans
            )

        # 人物が含まれるフレームの補完保存
        for idx, item in enumerate(batch_items, 1):
          if item["has_person"] and idx not in saved_indices:
            fn = self._save_frame_with_real_time(
                item["frame"],
                item["video_path"],
                item["sec"],
                save_folder,
                prefix="PERSON",
            )
            saved_indices.add(idx)
            self.log(f"  └ 👤 [人物保存] No.{idx:02d} -> {fn}")
            self._save_csv_log(
                save_folder, item["video_path"], item["sec"], "PERSON", ans
            )

      else:
        self.log(f"[API Error] Status: {response.status_code}")
    except Exception as e:
      self.log(f"[VLM Error] 通信エラー: {e}")

  # ------------------------------------------
  # ユーティリティ & 保存関数
  # ------------------------------------------
  def _save_frame_with_real_time(
      self, frame, video_path, sec, save_folder, prefix=""
  ):
    """動画のタイムスタンプに再生秒数を加算し、実時刻のファイル名で画像保存"""
    start_dt = self._parse_video_start_datetime(video_path)
    actual_dt = start_dt + timedelta(seconds=sec)

    datetime_str = actual_dt.strftime("%Y%m%d_%H%M%S")
    prefix_str = f"{prefix}_" if prefix else ""
    file_name = f"{prefix_str}{datetime_str}_{sec:.1f}s.jpg"

    save_path = os.path.join(save_folder, file_name)
    cv2.imwrite(save_path, frame)
    return file_name

  def _save_csv_log(self, save_folder, video_path, sec, event_type, ai_result):
    """検知イベントを CSV ファイルに1行追記"""
    csv_path = os.path.join(save_folder, "detection_log.csv")
    file_exists = os.path.exists(csv_path)

    start_dt = self._parse_video_start_datetime(video_path)
    actual_dt = start_dt + timedelta(seconds=sec)
    actual_time_str = actual_dt.strftime("%Y-%m-%d %H:%M:%S")

    with open(csv_path, mode="a", newline="", encoding="utf-8-sig") as f:
      writer = csv.writer(f)
      if not file_exists:
        writer.writerow([
            "実際の撮影日時",
            "動画ファイルパス",
            "動画内位置(秒)",
            "イベント種別",
            "AI解析概要",
        ])
      summary = (
          ai_result.replace("\n", " ")[:100] + "..."
          if len(ai_result) > 100
          else ai_result.replace("\n", " ")
      )
      writer.writerow([
          actual_time_str,
          video_path,
          f"{sec:.1f}",
          event_type,
          summary,
      ])

  def _calc_motion(self, prev, curr):
    """フレーム間差分（ガウシアンフィルタ＋2値化）による背景動きの計算"""
    if prev is None:
      return 1.0
    g1 = cv2.GaussianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    g2 = cv2.GaussianBlur(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    delta = cv2.absdiff(g1, g2)
    thresh = cv2.threshold(delta, 15, 255, cv2.THRESH_BINARY)[1]
    return np.sum(thresh > 0) / thresh.size

  def _calc_impact(
      self, res, shape, class_names=None, user_target_classes=None
  ):
    """検出物体の信頼度とバウンディングボックスの面積からインパクトスコアを計算"""
    h, w = shape[:2]
    area = w * h
    max_s, detected, has_p = 0.0, [], False
    default_cls = [0, 2, 3, 7, 15, 16, 17, 18, 19, 21]

    for box in res.boxes:
      c = int(box.cls[0])
      conf = float(box.conf[0])
      c_name = (
          class_names[c].lower()
          if class_names and c in class_names
          else str(c)
      )

      # ユーザー指定クラスと照合
      if user_target_classes:
        if c_name not in user_target_classes:
          continue
      else:
        if c not in default_cls:
          continue

      x1, y1, x2, y2 = box.xyxy[0].tolist()
      r = ((x2 - x1) * (y2 - y1)) / area

      # 人物の場合は重要度（重み）を高く計算
      if c == 0 or c_name == "person":
        if conf >= 0.35 and r >= 0.005:
          has_p = True
          max_s = max(max_s, r * conf * 4.0)
      else:
        if conf >= 0.45 and r >= 0.03:
          max_s = max(max_s, r * conf)

    return max_s, detected, has_p

  def _get_sorted_videos(self, folder):
    """指定フォルダ以下の動画ファイルを更新日時順（時系列）でソートして取得"""
    exts = (".mp4", ".avi", ".mkv", ".mov")
    files = []
    for r, _, fs in os.walk(folder):
      for f in fs:
        if f.lower().endswith(exts):
          p = os.path.join(r, f)
          files.append((os.path.getmtime(p), p))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]

  def _finish_process(self):
    """処理完了時のUI状態リセット"""
    self.is_running = False
    self.after(0, lambda: self.btn_start.configure(state="normal"))
    self.after(0, lambda: self.btn_stop.configure(state="disabled"))
    self.after(
        0,
        lambda: (
            self.lbl_progress_info.configure(
                text="=== すべての解析が完了しました ==="
            ),
            self.progress_bar.set(1.0),
        ),
    )


# ==========================================
# 4. アプリケーション エントリポイント
# ==========================================
if __name__ == "__main__":
  app = SurveillanceAIMonitor()
  app.mainloop()
