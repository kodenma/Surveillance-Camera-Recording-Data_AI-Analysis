import base64
from collections import deque
import csv
import ctypes
from datetime import datetime, timedelta
import importlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import cv2
import customtkinter as ctk
import numpy as np
import psutil
import requests
from ultralytics import YOLO

# ==========================================
# 1. UIテーマおよびグローバル定数の設定
# ==========================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

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


# ==========================================
# Windows スリープ防止用コンテキストマネージャー
# ==========================================
class PreventSleep:

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __enter__(self):
        if os.name == "nt":
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(
                    self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
                )
            except Exception:
                pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.name == "nt":
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
            except Exception:
                pass


# ==========================================
# 2. 検出対象クラス選択 サブウィンドウ (モーダル)
# ==========================================
class ClassSelectionWindow(ctk.CTkToplevel):

    def __init__(self, parent, selected_classes_list, on_apply_callback):
        super().__init__(parent)
        self.title("検出対象クラス 一覧設定")
        self.geometry("720x640")
        self.grab_set()

        self.selected_set = set([c.strip() for c in selected_classes_list])
        self.on_apply_callback = on_apply_callback
        self.checkboxes = {}

        all_coco = set()
        for cat_list in COCO_CLASSES_BY_CATEGORY.values():
            all_coco.update(cat_list)
        self.custom_initial = [c for c in selected_classes_list if c not in all_coco]

        self._create_widgets()

    def _create_widgets(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)

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

        scroll_frame = ctk.CTkScrollableFrame(main_frame)
        scroll_frame.pack(fill="both", expand=True, pady=10)

        for category, cls_list in COCO_CLASSES_BY_CATEGORY.items():
            cat_label = ctk.CTkLabel(
                scroll_frame, text=f"■ {category}", font=("メイリオ", 12, "bold")
            )
            cat_label.pack(anchor="w", pady=(10, 2))

            grid_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
            grid_frame.pack(fill="x", padx=10, pady=2)

            for i, cls_name in enumerate(cls_list):
                var = ctk.BooleanVar(value=(cls_name in self.selected_set))
                chk = ctk.CTkCheckBox(grid_frame, text=cls_name, variable=var)
                chk.grid(row=i // 3, column=i % 3, sticky="w", padx=10, pady=4)
                self.checkboxes[cls_name] = var

        custom_frame = ctk.CTkFrame(main_frame)
        custom_frame.pack(fill="x", pady=10, padx=5)

        ctk.CTkLabel(
            custom_frame,
            text="✍️ カスタムクラス追加 (独自モデル用・カンマ区切り):",
            font=("メイリオ", 11, "bold"),
        ).pack(anchor="w", padx=5, pady=2)

        self.entry_custom_classes = ctk.CTkEntry(
            custom_frame,
            placeholder_text="例: custom_person, helmet, danger_object",
            width=650,
        )
        self.entry_custom_classes.pack(fill="x", padx=5, pady=5)

        if self.custom_initial:
            self.entry_custom_classes.insert(0, ", ".join(self.custom_initial))

        btn_apply = ctk.CTkButton(
            main_frame,
            text="設定を適用して閉じる",
            fg_color="green",
            hover_color="darkgreen",
            command=self._apply_and_close,
        )
        btn_apply.pack(fill="x", pady=5)

    def _select_all(self):
        for var in self.checkboxes.values():
            var.set(True)

    def _deselect_all(self):
        for var in self.checkboxes.values():
            var.set(False)

    def _apply_and_close(self):
        res = [cls for cls, var in self.checkboxes.items() if var.get()]
        custom_text = self.entry_custom_classes.get().strip()
        if custom_text:
            custom_list = [c.strip() for c in custom_text.split(",") if c.strip()]
            for c in custom_list:
                if c not in res:
                    res.append(c)

        self.on_apply_callback(res)
        self.destroy()


# ==========================================
# 3. メインアプリケーション クラス
# ==========================================
class SurveillanceAIMonitor(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("防犯カメラ AI 異常検知システム (YOLO + LM Studio)")
        self.geometry("1020x980")

        self.is_running = False
        self.stop_requested = False
        self.analysis_thread = None

        self.selected_target_classes = ["person", "car", "motorcycle", "bicycle"]
        self.debug_log_path = os.path.join(
            os.getcwd(), "surveillance_ai_debug.log"
        )

        # アダプティブ閾値制御用の状態変数
        self.base_yolo_conf = 0.50
        self.current_yolo_conf = 0.60
        self.consecutive_normal_vlm_count = 0
        self.conf_history_window = 5
        self.conf_history = deque(
            [self.base_yolo_conf] * self.conf_history_window,
            maxlen=self.conf_history_window,
        )

        self.available_devices = self._detect_hardware_devices()

        # メインのタブビュー作成
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_main = self.tabview.add("🚀 メイン解析")
        self.tab_config = self.tabview.add("⚙️ 詳細設定")
        self.tab_info = self.tabview.add("ℹ️ 情報")

        self._create_main_tab_widgets()
        self._create_config_tab_widgets()
        self._create_info_tab_widgets()

        self._start_resource_monitoring()

    def _detect_hardware_devices(self):
        devices = {"CPU": "Intel CPU (OpenVINO)"}
        try:
            import openvino as ov

            core = ov.Core()
            ov_devs = [str(d).upper() for d in core.available_devices]
            devices["iGPU"] = (
                "Intel iGPU (OpenVINO)"
                if any("GPU" in d for d in ov_devs)
                else "Intel iGPU (OpenVINO対応)"
            )
            devices["NPU"] = (
                "Intel NPU (OpenVINO)"
                if any("NPU" in d for d in ov_devs)
                else "Intel NPU (OpenVINO対応)"
            )
        except Exception as e:
            self._write_debug_file(f"[Hardware Detect] OpenVINO init error: {e}")
            devices["iGPU"] = "Intel iGPU (OpenVINO)"
            devices["NPU"] = "Intel NPU (OpenVINO)"

        has_nvidia = False
        gpu_name_str = "NVIDIA dGPU"
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name_str = f"NVIDIA dGPU ({torch.cuda.get_device_name(0)})"
                has_nvidia = True
        except Exception:
            pass

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

        devices["dGPU"] = (
            gpu_name_str if has_nvidia else "NVIDIA dGPU (未検出/CUDA未設定)"
        )
        return devices

    # ------------------------------------------
    # タブ 1: メイン解析 UI
    # ------------------------------------------
    def _create_main_tab_widgets(self):
        parent = self.tab_main

        folder_frame = ctk.CTkFrame(parent)
        folder_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(folder_frame, text="動画フォルダ:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        self.entry_video_dir = ctk.CTkEntry(folder_frame, width=540)
        self.entry_video_dir.insert(0, r"未指定")
        self.entry_video_dir.grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkButton(
            folder_frame, text="参照", width=70, command=self._browse_video_dir
        ).grid(row=0, column=2, padx=5, pady=5)

        ctk.CTkLabel(folder_frame, text="保存フォルダ:").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        self.entry_save_dir = ctk.CTkEntry(folder_frame, width=540)
        self.entry_save_dir.insert(0, r"未指定")
        self.entry_save_dir.grid(row=1, column=1, padx=5, pady=5)
        ctk.CTkButton(
            folder_frame, text="参照", width=70, command=self._browse_save_dir
        ).grid(row=1, column=2, padx=5, pady=5)

        hw_frame = ctk.CTkFrame(parent)
        hw_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(hw_frame, text="推論デバイス:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        dev_options = list(self.available_devices.values())
        self.combo_device = ctk.CTkOptionMenu(
            hw_frame, values=dev_options, width=220
        )
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
                "自動",
                "yolo26s.pt",
                "yolo26n.pt",
                "yolo11s.pt",
                "yolov8s.pt",
            ],
            width=160,
        )
        self.combo_yolo_model.set("自動")
        self.combo_yolo_model.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        ctk.CTkButton(
            hw_frame,
            text="🤖 AI自動最適設定",
            width=130,
            fg_color="#8B5CF6",
            hover_color="#7C3AED",
            command=self._auto_config_by_ai,
        ).grid(row=0, column=4, padx=15, pady=5, sticky="e")

        class_frame = ctk.CTkFrame(parent)
        class_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            class_frame, text="検出対象クラス:", font=("メイリオ", 11, "bold")
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")

        ctk.CTkButton(
            class_frame,
            text="🎯 検出クラスを一覧・追加設定...",
            width=210,
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
            text="🐞 デバッグモード",
            variable=self.chk_debug_mode_var,
        ).grid(row=0, column=3, padx=15, pady=5, sticky="e")

        prompt_frame = ctk.CTkFrame(parent)
        prompt_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(prompt_frame, text="AI解析プロンプト:").pack(
            anchor="w", padx=5, pady=2
        )
        self.text_prompt = ctk.CTkTextbox(prompt_frame, height=80)
        self.text_prompt.insert(
            "1.0",
            "定点防犯カメラ画像の時系列解析を行ってください。\n"
            "【手順】\n"
            "1. 人物・車両・指定対象物等の動きや滞留を時系列で確認。\n"
            "2. 侵入・不審行為・長時間の滞留等の「異常」があるか判定。\n\n"
            "※必ず以下の形式で簡潔に出力してください（概要は60文字以内）：\n"
            "【判定】: 異常あり (または 異常なし)\n"
            "【異常フレーム】: 1, 5 (該当する画像の番号を半角数字カンマ区切りで列挙)\n"
            "【概要】: (簡潔な理由)",
        )
        self.text_prompt.pack(fill="x", padx=5, pady=5)

        ctrl_frame = ctk.CTkFrame(parent)
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

        log_frame = ctk.CTkFrame(parent)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.text_log = ctk.CTkTextbox(log_frame, font=("Consolas", 11))
        self.text_log.pack(fill="both", expand=True, padx=5, pady=5)

        self.status_bar = ctk.CTkLabel(
            parent,
            text="CPU: --% | RAM: --%",
            anchor="w",
            font=("Consolas", 11),
        )
        self.status_bar.pack(fill="x", side="bottom", padx=15, pady=3)

    # ------------------------------------------
    # タブ 2: 詳細設定 UI
    # ------------------------------------------
    def _create_config_tab_widgets(self):
        parent = self.tab_config

        scroll_frame = ctk.CTkScrollableFrame(parent)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # --- A. LM Studio連携 アダプティブ閾値設定 ---
        group_adaptive = ctk.CTkFrame(scroll_frame)
        group_adaptive.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            group_adaptive,
            text="⚡ LM Studio連携 アダプティブ閾値制御（定点カメラ高速化）",
            font=("メイリオ", 12, "bold"),
        ).pack(anchor="w", padx=10, pady=5)

        adaptive_grid = ctk.CTkFrame(group_adaptive, fg_color="transparent")
        adaptive_grid.pack(fill="x", padx=10, pady=5)

        self.chk_use_adaptive_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            adaptive_grid,
            text=(
                "LM Studioの判定結果を参考に、YOLOの閾値を動的に自動調整する (推奨)"
            ),
            variable=self.chk_use_adaptive_var,
            font=("メイリオ", 11, "bold"),
        ).grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(
            adaptive_grid,
            text=(
                "※『異常なし』判定が連続すると、無駄なVLM送信を減らすため"
                "YOLO Conf閾値を徐々に自動引き上げます。"
            ),
            text_color="gray",
        ).grid(row=1, column=0, columnspan=2, padx=5, pady=2, sticky="w")

        # --- B. 静止物体・無駄送信防止フィルタ設定 ---
        group_filter = ctk.CTkFrame(scroll_frame)
        group_filter.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            group_filter,
            text="🛑 静止物体・無駄送信除外フィルタ (駐車車両などの対策)",
            font=("メイリオ", 12, "bold"),
        ).pack(anchor="w", padx=10, pady=5)

        filter_grid = ctk.CTkFrame(group_filter, fg_color="transparent")
        filter_grid.pack(fill="x", padx=10, pady=5)

        self.chk_ignore_static_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            filter_grid,
            text="静止物体（駐車中の車・動かない物体）をVLM送信から除外する",
            variable=self.chk_ignore_static_var,
            font=("メイリオ", 11, "bold"),
        ).grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(
            filter_grid, text="物体移動量の最小判定閾値 (px):"
        ).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entry_min_move_dist = ctk.CTkEntry(filter_grid, width=100)
        self.entry_min_move_dist.insert(0, "15.0")
        self.entry_min_move_dist.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # --- C. YOLO 検出パラメータ設定 ---
        group_yolo = ctk.CTkFrame(scroll_frame)
        group_yolo.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            group_yolo,
            text="🔍 YOLO 検出初期パラメータ設定",
            font=("メイリオ", 12, "bold"),
        ).pack(anchor="w", padx=10, pady=5)

        yolo_grid = ctk.CTkFrame(group_yolo, fg_color="transparent")
        yolo_grid.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            yolo_grid, text="初期信頼度閾値 (Base Conf Threshold) [0.1 - 0.95]:"
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry_yolo_conf = ctk.CTkEntry(yolo_grid, width=100)
        self.entry_yolo_conf.insert(0, "0.50")
        self.entry_yolo_conf.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(
            yolo_grid, text="最小検出物体面積比 (Min Area Ratio) [0.001 - 0.1]:"
        ).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entry_min_area_ratio = ctk.CTkEntry(yolo_grid, width=100)
        self.entry_min_area_ratio.insert(0, "0.005")
        self.entry_min_area_ratio.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # --- D. サンプリング & 追跡設定 ---
        group_sample = ctk.CTkFrame(scroll_frame)
        group_sample.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            group_sample, text="⏱️ フレーム サンプリング設定", font=("メイリオ", 12, "bold")
        ).pack(anchor="w", padx=10, pady=5)

        sample_grid = ctk.CTkFrame(group_sample, fg_color="transparent")
        sample_grid.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(sample_grid, text="通常時サンプリング(秒):").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        self.entry_normal_sample = ctk.CTkEntry(sample_grid, width=100)
        self.entry_normal_sample.insert(0, "0.5")
        self.entry_normal_sample.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(
            sample_grid, text="検知時サンプリング(秒 / 0で全フレーム):"
        ).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entry_high_sample = ctk.CTkEntry(sample_grid, width=100)
        self.entry_high_sample.insert(0, "0.1")
        self.entry_high_sample.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(sample_grid, text="検知時追跡時間(秒):").grid(
            row=2, column=0, padx=5, pady=5, sticky="w"
        )
        self.entry_tracking_dur = ctk.CTkEntry(sample_grid, width=100)
        self.entry_tracking_dur.insert(0, "5.0")
        self.entry_tracking_dur.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # --- E. LM Studio (VLM) 連携設定 ---
        group_lm = ctk.CTkFrame(scroll_frame)
        group_lm.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            group_lm,
            text="🤖 LM Studio (VLM) 連携・送信設定",
            font=("メイリオ", 12, "bold"),
        ).pack(anchor="w", padx=10, pady=5)

        lm_grid = ctk.CTkFrame(group_lm, fg_color="transparent")
        lm_grid.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(lm_grid, text="LM Studio Endpoint URL:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        self.entry_lm_url = ctk.CTkEntry(lm_grid, width=350)
        self.entry_lm_url.insert(0, "http://localhost:1234/v1/chat/completions")
        self.entry_lm_url.grid(
            row=0, column=1, columnspan=2, padx=5, pady=5, sticky="w"
        )

        ctk.CTkButton(
            lm_grid,
            text="接続テスト",
            width=85,
            fg_color="gray",
            command=self._test_lm_connection,
        ).grid(row=0, column=3, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(lm_grid, text="VLMモデル名:").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        self.entry_lm_model = ctk.CTkEntry(lm_grid, width=350)
        self.entry_lm_model.insert(0, "google/gemma-4-e4b")
        self.entry_lm_model.grid(
            row=1, column=1, columnspan=2, padx=5, pady=5, sticky="w"
        )

        ctk.CTkButton(
            lm_grid,
            text="モデル自動取得",
            width=110,
            fg_color="#10B981",
            command=self._fetch_lm_models,
        ).grid(row=1, column=3, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(lm_grid, text="LM Studio 1回あたりの最大送信バッチ枚数:").grid(
            row=2, column=0, padx=5, pady=5, sticky="w"
        )
        self.entry_vlm_batch_size = ctk.CTkEntry(lm_grid, width=100)
        self.entry_vlm_batch_size.insert(0, "10")
        self.entry_vlm_batch_size.grid(
            row=2, column=1, padx=5, pady=5, sticky="w"
        )

        ctk.CTkLabel(lm_grid, text="VLM送信時 画像リサイズ最大長(px):").grid(
            row=3, column=0, padx=5, pady=5, sticky="w"
        )
        self.entry_max_img_size = ctk.CTkEntry(lm_grid, width=100)
        self.entry_max_img_size.insert(0, "800")
        self.entry_max_img_size.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(lm_grid, text="Max Tokens:").grid(
            row=4, column=0, padx=5, pady=5, sticky="w"
        )
        self.entry_max_tokens = ctk.CTkEntry(lm_grid, width=100)
        self.entry_max_tokens.insert(0, "4000")
        self.entry_max_tokens.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(lm_grid, text="デコードスレッド数:").grid(
            row=5, column=0, padx=5, pady=5, sticky="w"
        )
        self.entry_threads = ctk.CTkEntry(lm_grid, width=100)
        self.entry_threads.insert(0, "4")
        self.entry_threads.grid(row=5, column=1, padx=5, pady=5, sticky="w")

    # ------------------------------------------
    # タブ 3: 情報 UI
    # ------------------------------------------
    def _create_info_tab_widgets(self):
        parent = self.tab_info

        info_frame = ctk.CTkFrame(parent)
        info_frame.pack(fill="both", expand=True, padx=20, pady=20)

        title_label = ctk.CTkLabel(
            info_frame,
            text="防犯カメラ AI 異常検知システム\n(YOLO + LM Studio VLM)",
            font=("メイリオ", 16, "bold"),
        )
        title_label.pack(pady=(20, 15))

        details = [
            ("👤 制作者", "kodenma (日本人)"),
            ("📜 ライセンス", "AGPL-3.0 License"),
            (
                "🔗 GitHub",
                "https://github.com/kodenma/Surveillance-Camera-Recording-Data_AI-Analysis",
            ),
        ]

        for label_text, val_text in details:
            row_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
            row_frame.pack(fill="x", padx=40, pady=8)

            ctk.CTkLabel(
                row_frame,
                text=f"{label_text}:",
                width=120,
                anchor="w",
                font=("メイリオ", 12, "bold"),
            ).pack(side="left")

            if "http" in val_text:
                entry_url = ctk.CTkEntry(row_frame, width=500)
                entry_url.insert(0, val_text)
                entry_url.configure(state="readonly")
                entry_url.pack(side="left", fill="x", expand=True)
            else:
                ctk.CTkLabel(
                    row_frame, text=val_text, anchor="w", font=("メイリオ", 12)
                ).pack(side="left")

        desc_text = (
            "\n【概要】\n"
            "本アプリケーションは、ローカルPC上で高速なYOLO物体検出モデルと"
            "LM Studio等のローカルVLM (Vision-Language Model) を連携させ、"
            "長時間の防犯カメラ映像から不審行為や侵入などの異常イベントを自動抽出・要約解析するシステムです。"
        )
        lbl_desc = ctk.CTkLabel(
            info_frame,
            text=desc_text,
            justify="left",
            anchor="w",
            font=("メイリオ", 11),
        )
        lbl_desc.pack(fill="x", padx=40, pady=20)

    # ------------------------------------------
    # ログ＆ユーティリティ
    # ------------------------------------------
    def _write_debug_file(self, msg):
        try:
            with open(self.debug_log_path, "a", encoding="utf-8") as f:
                f.write(f"{msg}\n")
        except Exception:
            pass

    def _debug_log(self, msg):
        if self.chk_debug_mode_var.get():
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self._write_debug_file(f"[{now_str}] [DEBUG] {msg}")

    def log(self, message):

        def _append():
            self.text_log.insert(
                "end", f"[{time.strftime('%H:%M:%S')}] {message}\n"
            )
            self.text_log.see("end")

        self.after(0, _append)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self._write_debug_file(f"[{now_str}] [INFO] {message}")

    def _open_class_selection_dialog(self):

        def on_apply(new_classes):
            self.selected_target_classes = new_classes
            summary_str = ", ".join(new_classes[:4])
            if len(new_classes) > 4:
                summary_str += f" (+{len(new_classes)-4}個)"
            self.lbl_selected_classes_summary.configure(
                text=f"選択中: {summary_str if new_classes else '全クラス対象'}"
            )
            self.log(
                f"🎯 検出対象クラスを更新しました ({len(new_classes)}個:"
                f" {', '.join(new_classes)})"
            )

        ClassSelectionWindow(self, self.selected_target_classes, on_apply)

    def _browse_video_dir(self):
        path = ctk.filedialog.askdirectory()
        if path:
            self.entry_video_dir.delete(0, "end")
            self.entry_video_dir.insert(0, path)

    def _browse_save_dir(self):
        path = ctk.filedialog.askdirectory()
        if path:
            self.entry_save_dir.delete(0, "end")
            self.entry_save_dir.insert(0, path)

    def _fetch_lm_models(self):
        url = (
            self.entry_lm_url.get().strip().replace("/chat/completions", "/models")
        )
        self.log(f"LM Studio からモデル一覧を取得中: {url}")

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
            except Exception as e:
                self.log(f"❌ モデル取得エラー: {e}")

        threading.Thread(target=fetch, daemon=True).start()

    def _auto_config_by_ai(self):
        video_dir = self.entry_video_dir.get().strip()
        exts = (".mp4", ".avi", ".mkv", ".mov")
        sample_video = None
        try:
            for root, _, files in os.walk(video_dir):
                for f in files:
                    if f.lower().endswith(exts):
                        sample_video = os.path.join(root, f)
                        break
                if sample_video:
                    break
        except Exception:
            pass

        cpu_cores = psutil.cpu_count(logical=True) or 4
        if sample_video and os.path.exists(sample_video):
            cap = cv2.VideoCapture(sample_video)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            else:
                w, h = 1920, 1080
        else:
            w, h = 1920, 1080

        rec_threads = max(2, min(8, cpu_cores - 2 if cpu_cores > 4 else 2))
        rec_normal = 0.5 if (w * h) <= (1920 * 1080) else 1.0

        self.entry_normal_sample.delete(0, "end")
        self.entry_normal_sample.insert(0, str(rec_normal))
        self.entry_high_sample.delete(0, "end")
        self.entry_high_sample.insert(0, "0.1")
        self.entry_tracking_dur.delete(0, "end")
        self.entry_tracking_dur.insert(0, "5.0")
        self.entry_threads.delete(0, "end")
        self.entry_threads.insert(0, str(rec_threads))
        self.log(f"✨ 自動最適設定を反映しました（解像度 {w}x{h} 用パラメータ）")

    def _test_lm_connection(self):
        url = self.entry_lm_url.get().strip()

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
                    self.log("✅ LM Studio 接続成功！")
                else:
                    self.log(f"⚠️ 接続失敗 Status: {res.status_code}")
            except Exception as e:
                self.log(f"❌ 接続エラー: {e}")

        threading.Thread(target=test, daemon=True).start()

    def _start_resource_monitoring(self):

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
        match = re.search(r"(\d{8})[\\/](\d{2})[\\/](\d{2})\.mp4$", file_path)
        if match:
            date_str, hour_str, min_str = match.groups()
            try:
                return datetime.strptime(
                    f"{date_str}{hour_str}{min_str}", "%Y%m%d%H%M"
                )
            except ValueError:
                pass
        try:
            return datetime.fromtimestamp(os.path.getmtime(file_path))
        except Exception:
            return datetime.now()

    def _resolve_and_load_yolo_model(self, model_input_str, selected_dev_text):
        if model_input_str == "自動":
            if os.path.exists("best.pt"):
                model_input_str = "best.pt"
                self.log("🤖 『自動』判定: カスタム学習モデル (best.pt) を使用します。")
            else:
                model_input_str = "yolo26s.pt"
                self.log("🤖 『自動』判定: 標準モデル (yolo26s.pt) を使用します。")

        self.log(f"📦 YOLOモデルをロード中: {model_input_str}")

        is_nvidia_dgpu = "dGPU" in selected_dev_text and "未検出" not in selected_dev_text
        use_openvino = not is_nvidia_dgpu

        pt_path = (
            model_input_str if os.path.exists(model_input_str) else f"{os.path.splitext(model_input_str)[0]}.pt"
        )

        try:
            pt_model = YOLO(pt_path)

            if use_openvino:
                self.log("⚙️ 設定に合わせて OpenVINO モデルを都度変換・更新しています...")
                try:
                    exported_path = pt_model.export(
                        format="openvino", int8=True, imgsz=640, half=False
                    )
                    model = YOLO(str(exported_path), task="detect")
                    self.log("✨ OpenVINO モデルの作成・ロードに成功しました")
                    return model, True  # OpenVINOモデルフラグ: True
                except Exception as ex:
                    self.log(f"⚠️ OpenVINO変換失敗 (PyTorch CPU推論にフォールバック): {ex}")
                    return pt_model, False  # OpenVINOモデルフラグ: False

            return pt_model, False

        except Exception as e:
            err_msg = f"❌ モデルのロードに失敗しました ({model_input_str}): {e}"
            self.log(err_msg)
            raise RuntimeError(err_msg)

    def _start_analysis(self):
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
        if self.is_running:
            self.stop_requested = True
            self.log("[Info] 停止要求を送信しました...")

    # ------------------------------------------
    # LM Studio判定結果に基づくアダプティブ閾値調整（移動平均制御）
    # ------------------------------------------
    def _update_adaptive_threshold(self, is_abnormal):
        if not self.chk_use_adaptive_var.get():
            return

        last_target = (
            self.conf_history[-1] if self.conf_history else self.base_yolo_conf
        )

        if is_abnormal:
            self.consecutive_normal_vlm_count = 0
            target_conf = max(self.base_yolo_conf, round(last_target - 0.1, 2))
            self.conf_history.append(target_conf)
        else:
            self.consecutive_normal_vlm_count += 1
            if self.consecutive_normal_vlm_count >= 2:
                target_conf = min(0.85, round(last_target + 0.05, 2))
            else:
                target_conf = last_target
            self.conf_history.append(target_conf)

        ma_conf = round(sum(self.conf_history) / len(self.conf_history), 2)
        applied_conf = min(0.85, round(ma_conf + 0.1, 2))

        if applied_conf != self.current_yolo_conf:
            status_str = "異常あり" if is_abnormal else "異常なし"
            self.log(
                f"⚡ [アダプティブ制御] AI判定『{status_str}』 -> YOLO Conf "
                f"適用閾値(平均+0.1): {self.current_yolo_conf:.2f} -> {applied_conf:.2f} "
                f"(移動平均: {ma_conf:.2f}, 目標: {target_conf:.2f})"
            )
            self.current_yolo_conf = applied_conf

    # ------------------------------------------
    # メイン解析 パイプライン
    # ------------------------------------------
    def _run_analysis_process(self):
        with PreventSleep():
            video_dir = self.entry_video_dir.get().strip()
            save_dir = self.entry_save_dir.get().strip()
            model_name = self.combo_yolo_model.get().strip()
            selected_dev_text = self.combo_device.get()

            # パラメータ読み込み
            try:
                self.base_yolo_conf = float(self.entry_yolo_conf.get().strip())
            except ValueError:
                self.base_yolo_conf = 0.50

            self.consecutive_normal_vlm_count = 0
            self.conf_history = deque(
                [self.base_yolo_conf] * self.conf_history_window,
                maxlen=self.conf_history_window,
            )
            self.current_yolo_conf = min(0.85, round(self.base_yolo_conf + 0.1, 2))

            try:
                min_area_ratio = float(self.entry_min_area_ratio.get().strip())
            except ValueError:
                min_area_ratio = 0.005

            ignore_static = self.chk_ignore_static_var.get()
            try:
                min_move_dist = float(self.entry_min_move_dist.get().strip())
            except ValueError:
                min_move_dist = 15.0

            try:
                vlm_batch_size = int(self.entry_vlm_batch_size.get().strip())
            except ValueError:
                vlm_batch_size = 10

            try:
                max_img_size = int(self.entry_max_img_size.get().strip())
            except ValueError:
                max_img_size = 800

            lm_url = self.entry_lm_url.get().strip()
            lm_model = self.entry_lm_model.get().strip()
            prompt_text = self.text_prompt.get("1.0", "end").strip()

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

            # YOLOモデルロード（毎回都度生成）
            try:
                yolo_model, is_openvino_model = self._resolve_and_load_yolo_model(
                    model_name, selected_dev_text
                )
            except Exception:
                self._finish_process()
                return

            # OpenVINOモデルかPyTorchモデルかに応じて device 引数を切り替え
            if is_openvino_model:
                if "iGPU" in selected_dev_text:
                    target_device = "GPU"
                elif "NPU" in selected_dev_text:
                    target_device = "NPU"
                else:
                    target_device = "CPU"
            else:
                if "dGPU" in selected_dev_text and "未検出" not in selected_dev_text:
                    target_device = 0
                else:
                    target_device = "cpu"

            video_files = self._get_sorted_videos(video_dir)
            total_videos = len(video_files)
            if total_videos == 0:
                self.log("⚠️ 対象フォルダ内に動画ファイルが見つかりませんでした。")
                self._finish_process()
                return

            self.log(
                f"[Info] 計 {total_videos} 件の動画をスキャンします。"
                f" (推論デバイス: {selected_dev_text} -> device='{target_device}', "
                f"OpenVINO化: {is_openvino_model}, "
                f"初期Base Conf: {self.base_yolo_conf}, 初期適用閾値: {self.current_yolo_conf})"
            )

            frame_queue = queue.Queue(maxsize=200)
            video_queue = queue.Queue()
            for vp in video_files:
                video_queue.put(vp)

            high_freq_until = {}

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

                        step = (
                            1
                            if active_interval <= 0.0
                            else max(1, int(fps * active_interval))
                        )
                        target_idx += step

                    cap.release()
                    video_queue.task_done()

            decoder_threads = []
            for _ in range(num_threads):
                t = threading.Thread(target=decoder_worker, daemon=True)
                t.start()
                decoder_threads.append(t)

            vlm_buffer = []
            last_collected = {}
            prev_frames = {}
            last_boxes = {}
            processed_videos_set = set()

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

                self.after(
                    0,
                    lambda c=curr_v_count, t=total_videos, p=pct, name=v_name: (
                        self.lbl_progress_info.configure(
                            text=f"[{c} / {t} 件 ({p:.1f}%)] 処理中: {name}"
                        ),
                        self.progress_bar.set(p / 100.0),
                    ),
                )

                # 正確な device 引数で YOLO 推論を実行
                res = yolo_model(
                    frame,
                    device=target_device,
                    conf=self.current_yolo_conf,
                    imgsz=640,
                    verbose=False,
                )[0]

                prev_box_dict = last_boxes.get(vp, {})
                impact_score, trigger_desc, has_target, is_moving, current_box_dict = (
                    self._calc_impact_and_movement(
                        res,
                        frame.shape,
                        yolo_model.names,
                        target_classes,
                        self.current_yolo_conf,
                        min_area_ratio,
                        prev_box_dict,
                        min_move_dist,
                    )
                )
                last_boxes[vp] = current_box_dict

                if impact_score > 0:
                    prev_f = prev_frames.get(vp, None)
                    motion_score = self._calc_motion(prev_f, frame)
                    last_s = last_collected.get(vp, -1.0)

                    # 静止物体除外判定
                    should_skip_static = (
                        ignore_static and (not is_moving) and (motion_score < 0.01)
                    )

                    if (
                        (motion_score > 0.005 or has_target)
                        and (sec - last_s >= 1.0)
                        and (not should_skip_static)
                    ):
                        high_freq_until[vp] = sec + tracking_duration
                        last_collected[vp] = sec

                        full_trigger_str = (
                            f"{trigger_desc} | 画面動きスコア: {motion_score:.4f}"
                        )
                        vlm_buffer.append({
                            "frame": frame,
                            "video_path": vp,
                            "sec": sec,
                            "has_person": has_target,
                            "impact_score": impact_score,
                            "trigger_reason": full_trigger_str,
                        })

                        self.log(
                            f"🎯 [トリガー検知] {v_name} ({sec:.1f}s) ->"
                            f" {full_trigger_str}"
                        )

                        if len(vlm_buffer) >= vlm_batch_size:
                            self._process_vlm_batch(
                                vlm_buffer[:vlm_batch_size],
                                lm_url,
                                lm_model,
                                save_dir,
                                prompt_text,
                                max_tokens,
                                max_img_size,
                            )
                            del vlm_buffer[:vlm_batch_size]

                    elif should_skip_static:
                        self._debug_log(
                            f"[静止スキップ] {v_name} ({sec:.1f}s):"
                            " 物体が静止しているためVLM送信をスキップ。"
                        )

                prev_frames[vp] = frame
                frame_queue.task_done()

            if vlm_buffer and not self.stop_requested:
                self.log(f"[Info] 残り {len(vlm_buffer)} 枚を最終解析します。")
                self._process_vlm_batch(
                    vlm_buffer,
                    lm_url,
                    lm_model,
                    save_dir,
                    prompt_text,
                    max_tokens,
                    max_img_size,
                )
                vlm_buffer.clear()

            self.log(
                "=== 解析が完了しました ==="
                if not self.stop_requested
                else "=== 処理を停止しました ==="
            )
            self._finish_process()

    # ------------------------------------------
    # ターゲットスコア判定 ＆ 物体移動距離トラッキング
    # ------------------------------------------
    def _calc_impact_and_movement(
        self,
        res,
        shape,
        class_names=None,
        user_target_classes=None,
        conf_thresh=0.50,
        min_area_ratio=0.005,
        prev_box_dict=None,
        min_move_dist=15.0,
    ):
        h, w = shape[:2]
        area = w * h
        max_s = 0.0
        has_target = False
        is_moving = False

        target_set = (
            set([c.strip().lower() for c in user_target_classes])
            if user_target_classes
            else None
        )
        current_box_dict = {}
        trigger_details = []

        for box in res.boxes:
            c = int(box.cls[0])
            conf = float(box.conf[0])

            if class_names and c in class_names:
                c_name = str(class_names[c]).lower()
            else:
                c_name = str(c).lower()

            if target_set and (c_name not in target_set):
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            r = ((x2 - x1) * (y2 - y1)) / area

            current_box_dict[c_name] = (cx, cy)

            if conf >= conf_thresh and r >= min_area_ratio:
                has_target = True

                move_dist = 0.0
                if prev_box_dict and c_name in prev_box_dict:
                    pcx, pcy = prev_box_dict[c_name]
                    move_dist = math.hypot(cx - pcx, cy - pcy)
                    if move_dist >= min_move_dist:
                        is_moving = True
                else:
                    is_moving = True
                    move_dist = min_move_dist

                weight = 4.0 if c_name in ["person", "human"] else 2.0
                score = r * conf * weight
                max_s = max(max_s, score)

                trigger_details.append(
                    f"{c_name}(conf:{conf:.2f}, 面積比:{r*100:.2f}%, 移動:{move_dist:.1f}px)"
                )

        trigger_desc = (
            ", ".join(trigger_details) if trigger_details else "対象なし"
        )
        return max_s, trigger_desc, has_target, is_moving, current_box_dict

    # ------------------------------------------
    # VLM (Vision-Language Model) 送信・解析・保存
    # ------------------------------------------
    def _process_vlm_batch(
        self,
        batch_items,
        url,
        model_id,
        save_folder,
        prompt_text,
        max_tokens,
        max_img_size=800,
    ):
        self.log(
            f"🚀 【AI解析送信】 {len(batch_items)} 枚のトリガー付きフレームを AI"
            f" ({model_id}) へ送信中..."
        )

        batch_items.sort(key=lambda x: (x["video_path"], x["sec"]))

        content_list = [{"type": "text", "text": prompt_text}]
        for idx, item in enumerate(batch_items, 1):
            h, w = item["frame"].shape[:2]
            if max(h, w) > max_img_size:
                scale = max_img_size / float(max(h, w))
                resized = cv2.resize(
                    item["frame"],
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                resized = item["frame"]

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

                is_abnormal = (
                    "【判定】: 異常あり" in ans
                    or "【判定】:異常あり" in ans
                    or "【判定】: 异常あり" in ans
                )

                self._update_adaptive_threshold(is_abnormal)

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
                                int(n)
                                for n in found_nums
                                if 1 <= int(n) <= len(batch_items)
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
                        self.log(
                            f"  └ 💾 [異常保存] No.{idx:02d} -> {fn} (トリガー:"
                            f" {item['trigger_reason']})"
                        )
                        self._save_csv_log(
                            save_folder,
                            item["video_path"],
                            item["sec"],
                            "ABNORMAL",
                            item["trigger_reason"],
                            ans,
                        )
                else:
                    self.log(
                        "  └ ℹ️ AI判定: 『異常なし』のため、画像保存をスキップしました。"
                    )

            else:
                self.log(f"[API Error] Status: {response.status_code}")
        except Exception as e:
            self.log(f"[VLM Error] 通信エラー: {e}")

    def _save_frame_with_real_time(
        self, frame, video_path, sec, save_folder, prefix=""
    ):
        start_dt = self._parse_video_start_datetime(video_path)
        actual_dt = start_dt + timedelta(seconds=sec)

        datetime_str = actual_dt.strftime("%Y%m%d_%H%M%S")
        prefix_str = f"{prefix}_" if prefix else ""
        file_name = f"{prefix_str}{datetime_str}_{sec:.1f}s.jpg"

        save_path = os.path.join(save_folder, file_name)
        cv2.imwrite(save_path, frame)
        return file_name

    def _save_csv_log(
        self, save_folder, video_path, sec, event_type, trigger_reason, ai_result
    ):
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
                    "検知トリガー詳細",
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
                trigger_reason,
                summary,
            ])

    def _calc_motion(self, prev, curr):
        if prev is None:
            return 1.0
        g1 = cv2.GaussianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        g2 = cv2.GaussianBlur(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        delta = cv2.absdiff(g1, g2)
        thresh = cv2.threshold(delta, 15, 255, cv2.THRESH_BINARY)[1]
        return np.sum(thresh > 0) / thresh.size

    def _get_sorted_videos(self, folder):
        exts = (".mp4", ".avi", ".mkv", ".mov")
        files = []

        def _scan(target_dir):
            try:
                with os.scandir(target_dir) as entries:
                    for entry in entries:
                        if entry.is_dir(follow_symlinks=False):
                            _scan(entry.path)
                        elif entry.is_file(
                            follow_symlinks=False
                        ) and entry.name.lower().endswith(exts):
                            files.append((entry.stat().st_mtime, entry.path))
            except PermissionError:
                pass

        _scan(folder)
        files.sort(key=lambda x: x[0])
        return [p for _, p in files]

    def _finish_process(self):
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
# 4. エントリポイント
# ==========================================
if __name__ == "__main__":
    app = SurveillanceAIMonitor()
    app.mainloop()