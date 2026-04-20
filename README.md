# realtime-audio-record-transcript

即時麥克風錄音 → OpenAI Whisper 語音轉文字 → GPT 翻譯成繁體中文 → 自動寫入 Google Docs / Google Sheets。

適合用於現場活動、會議、演講或直播影片的即時逐字稿與同步翻譯。

## 功能

- **智慧 VAD 錄音**：偵測靜音自動切段，避免切到句中；可設定最短/最長錄音時間
- **Whisper 轉錄**：使用 OpenAI `whisper-1`，支援多語言，自動過濾幻覺（hallucination）重複句
- **GPT 翻譯**：可自訂 system prompt 與術語表，翻成繁體中文
- **Google Docs 輸出**：原文與譯文分別寫入不同 tab
- **Google Sheets 輸出**（選用）：時間戳 + 逐段文字寫入試算表
- **Watchdog**：錄音與轉錄執行緒崩潰時自動重啟
- **多任務設定**：透過 YAML 設定檔切換不同活動場景

## 架構

```
app.py          主程式（錄音、轉錄、翻譯、輸出的協調）
recording.py    PyAudio 麥克風錄音 + VAD 靜音切段
whisper_srt.py  Whisper API 呼叫 + hallucination 過濾
tasks/          任務設定檔（*.yaml，gitignore 不上傳）
temp/           錄音暫存（自動清除）
failed/         轉錄失敗的音檔
logs/           執行日誌（自動 rotate）
```

## 安裝

**需求**：Python 3.12、ffmpeg、PortAudio、[BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole)

### BlackHole 2ch 設定

你也可以透過 **BlackHole 2ch** 虛擬音訊裝置擷取系統聲音（例如會議軟體、瀏覽器播放的音訊），而非只限於實體麥克風錄製。

1. 安裝 BlackHole 2ch：
    ```bash
    brew install blackhole-2ch
    ```
2. 開啟 macOS **音訊 MIDI 設定**（`/Applications/Utilities/Audio MIDI Setup.app`）
3. 點擊左下角 `+` → **建立多重輸出裝置**，勾選：
    - 你的實體喇叭/耳機
    - BlackHole 2ch
4. 在**系統設定 → 聲音 → 輸出**，選擇剛建立的多重輸出裝置
5. 啟動 `app.py` 後，在裝置列表中選擇 **BlackHole 2ch** 作為輸入來源

這樣可在不影響正常音訊輸出的情況下，同時錄製系統播放的聲音。

```bash
# macOS
brew install ffmpeg portaudio

# 建立虛擬環境
python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## 設定

### 1. 環境變數

複製並填入 `.env`：

```bash
OPENAI_API_KEY=sk-...
serviceAccountFile=gcp-service.json
```

### 2. Google Cloud 服務帳戶

1. 在 Google Cloud Console 建立服務帳戶，下載 JSON 金鑰存為 `gcp-service.json`
2. 啟用 **Google Docs API** 與（選用）**Google Sheets API**
3. 將服務帳戶 email 加入目標 Google Doc / Sheet 的編輯權限

### 3. 任務設定檔

複製範例並依活動調整：

```bash
cp tasks/mcp.yaml.example tasks/my-event.yaml
```

主要參數說明（`tasks/*.yaml`）：

| 區塊          | 參數                    | 說明                           |
| ------------- | ----------------------- | ------------------------------ |
| `recording`   | `min_duration`          | 最短錄音秒數（預設 25s）       |
| `recording`   | `max_duration`          | 強制切斷秒數（預設 60s）       |
| `recording`   | `silence_ms`            | 觸發切斷的靜音毫秒數           |
| `recording`   | `silence_rms_threshold` | RMS 能量門檻（0–32767）        |
| `whisper`     | `language`              | BCP-47 語言碼（如 `en`、`zh`） |
| `whisper`     | `prompt`                | Whisper 提示詞，列出專有名詞   |
| `whisper`     | `min_dbfs`              | 低於此音量跳過轉錄             |
| `translation` | `model`                 | 翻譯模型（預設 `gpt-4o-mini`） |
| `translation` | `system_prompt`         | 翻譯 system prompt（含術語表） |
| `gdoc`        | `doc_id`                | Google Doc ID                  |
| `gdoc`        | `transcript_tab_id`     | 原文 tab ID                    |
| `gdoc`        | `translation_tab_id`    | 譯文 tab ID                    |
| `sheets`      | `worksheet_id`          | Google Sheet ID（留空跳過）    |

## 執行

```bash
python app.py
```

啟動後：

1. 顯示 `tasks/` 中的 YAML 清單，選擇任務編號
2. 顯示輸出目標（Google Doc ID、tab）
3. 輸入 Google Sheet tab 名稱（可略過）
4. 選擇錄音裝置（列出所有可用輸入裝置）
5. 開始即時錄音 → 轉錄 → 翻譯 → 寫入

按 `Ctrl+C` 結束。

## 流程說明

```
麥克風
  └─▶ recording.py（VAD 切段）
        └─▶ temp/*.mp3
              └─▶ whisper_srt.py（Whisper API 轉錄）
                    └─▶ translate_to_zhtw()（GPT 翻譯）
                          ├─▶ Google Docs（原文 + 譯文）
                          └─▶ Google Sheets（時間戳 + 原文）
```

每個步驟最多重試 3 次，超時後移至 `failed/` 並繼續處理下一段。

## 注意事項

- `tasks/*.yaml` 已列入 `.gitignore`，避免 doc_id、prompt 等敏感資訊外洩
- `gcp-service.json` 與 `.env` 同樣不應上傳到版本控制
- `logs/` 目錄每日自動建立，單檔最大 10MB，保留 5 份備份

## 授權

MIT
