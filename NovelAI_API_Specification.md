# NovelAI API 仕様書

このドキュメントは、NovelAI画像生成APIの仕様と実装時の注意点をまとめたものです。

## 目次
1. [基本情報](#基本情報)
2. [モデル](#モデル)
3. [サンプラー](#サンプラー)
4. [V4/V4.5 プロンプト構造](#v4v45-プロンプト構造)
5. [キャラクタープロンプト](#キャラクタープロンプト)
6. [座標システム](#座標システム)
7. [リクエスト形式](#リクエスト形式)
8. [レスポンス形式](#レスポンス形式)
9. [実装時の注意点](#実装時の注意点)

---

## 基本情報

### エンドポイント
- **ベースURL**: `https://image.novelai.net`
- **画像生成**: `/ai/generate-image` (POST)
- **ストリーミング生成**: `/ai/generate-image-stream` (POST)
- **画像拡張**: `/ai/augment-image` (POST)

### 認証
- **方式**: Bearer Token
- **ヘッダー**: `Authorization: Bearer {API_KEY}`

---

## モデル

### 利用可能なモデル

| モデルID | 説明 | バージョン |
|---------|------|-----------|
| `nai-diffusion-4-5-full` | V4.5 Full - 完全データセット | 最新 |
| `nai-diffusion-4-5-curated` | V4.5 Curated - 高品質データセット（推奨） | 最新 |
| `nai-diffusion-4-full` | V4 Full - 完全データセット | V4 |
| `nai-diffusion-3` | V3 - 安定版 | V3 |
| `nai-diffusion-2` | V2 | V2 |
| `nai-diffusion` | V1 | V1 |

### モデルの特徴

**V4/V4.5シリーズ:**
- 完全オリジナルモデル（Stable Diffusionベースではない）
- マルチキャラクター対応
- 構造化プロンプト形式（`v4_prompt`）
- 座標ベースのキャラクター配置
- 自然言語理解の向上

**V3以前:**
- 従来のテキストプロンプト形式
- `negative_prompt`を直接指定

---

## サンプラー

### 標準サンプラー（全モデル対応）

| サンプラー名 | 説明 |
|------------|------|
| `k_euler` | Euler法 |
| `k_euler_ancestral` | Euler Ancestral法 |
| `k_dpm_2` | DPM 2次 |
| `k_dpm_2_ancestral` | DPM 2次 Ancestral |
| `k_lms` | Linear Multi-Step |
| `k_dpmpp_2s_ancestral` | DPM++ 2S Ancestral |
| `k_dpmpp_sde` | DPM++ SDE |
| `k_dpmpp_2m` | DPM++ 2M |
| `k_dpmpp_2m_sde` | DPM++ 2M SDE |
| `k_dpmpp_3m_sde` | DPM++ 3M SDE |
| `k_dpm_adaptive` | DPM Adaptive |
| `k_dpm_fast` | DPM Fast |
| `plms` | PLMS |
| `ddim` | DDIM |
| `ddim_v3` | DDIM V3 |

### V4/V4.5専用サンプラー

| サンプラー名 | 内部処理 | 説明 |
|------------|---------|------|
| `nai_smea` | `k_euler_ancestral` + `sm=true` | SMEA (Sample Masking and Entropy Adaptive) |
| `nai_smea_dyn` | `k_euler_ancestral` + `sm=true` + `sm_dyn=true` | SMEA Dynamic |

**注意**: `nai_smea`と`nai_smea_dyn`は、実際のAPIリクエストでは`k_euler_ancestral`サンプラーと`sm`/`sm_dyn`フラグの組み合わせに変換されます。

---

## V4/V4.5 プロンプト構造

### 基本構造

V4/V4.5モデルでは、従来の`negative_prompt`ではなく、構造化された`v4_prompt`と`v4_negative_prompt`を使用します。

```json
{
  "v4_prompt": {
    "use_coords": false,
    "use_order": false,
    "caption": {
      "base_caption": "メインプロンプト",
      "char_captions": []
    }
  },
  "v4_negative_prompt": {
    "caption": {
      "base_caption": "メインネガティブプロンプト",
      "char_captions": []
    }
  }
}
```

### フィールド説明

**v4_prompt:**
- `use_coords` (boolean): 座標ベースの配置を使用するか
- `use_order` (boolean): 順序ベースの配置を使用するか
- `caption.base_caption` (string): メインプロンプト（シーン全体の説明）
- `caption.char_captions` (array): キャラクタープロンプトの配列

**v4_negative_prompt:**
- `caption.base_caption` (string): メインネガティブプロンプト
- `caption.char_captions` (array): キャラクター固有のネガティブプロンプト配列

---

## キャラクタープロンプト

### CharCaption 構造

```json
{
  "char_caption": "キャラクターの説明",
  "centers": [
    {
      "x": 0.1,
      "y": 0.1
    }
  ]
}
```

### フィールド説明

- `char_caption` (string): キャラクターのプロンプト
- `centers` (array): キャラクターの位置座標の配列
  - 座標を指定する場合: `[{"x": 0.1, "y": 0.1}]`
  - AUTO配置の場合: フィールド自体を省略（空配列ではない）

### 重要な注意点

1. **位置は必須**: V4/V4.5でキャラクタープロンプトを使用する場合、各キャラクターに位置を指定する必要があります
2. **AUTO位置は不可**: `centers`を空配列にしたり省略すると500エラーが発生します
3. **配列の数を一致**: `v4_prompt`と`v4_negative_prompt`の`char_captions`配列の要素数は同じである必要があります
4. **ネガティブプロンプトも位置を指定**: ネガティブプロンプトのchar_captionsにも同じ座標を設定します

### 正しい例

```json
{
  "v4_prompt": {
    "use_coords": true,
    "use_order": false,
    "caption": {
      "base_caption": "fantasy scene",
      "char_captions": [
        {
          "char_caption": "1girl, blonde hair",
          "centers": [{"x": 0.1, "y": 0.1}]
        },
        {
          "char_caption": "1boy, black hair",
          "centers": [{"x": 0.9, "y": 0.9}]
        }
      ]
    }
  },
  "v4_negative_prompt": {
    "caption": {
      "base_caption": "bad anatomy",
      "char_captions": [
        {
          "char_caption": "extra fingers",
          "centers": [{"x": 0.1, "y": 0.1}]
        },
        {
          "char_caption": "bad hands",
          "centers": [{"x": 0.9, "y": 0.9}]
        }
      ]
    }
  }
}
```

---

## 座標システム

### 5x5 グリッド

NovelAIは画像を5x5のグリッドに分割し、各セルに座標を割り当てます。

```
     1     2     3     4     5
A  (0.1,0.1) (0.1,0.3) (0.1,0.5) (0.1,0.7) (0.1,0.9)
B  (0.3,0.1) (0.3,0.3) (0.3,0.5) (0.3,0.7) (0.3,0.9)
C  (0.5,0.1) (0.5,0.3) (0.5,0.5) (0.5,0.7) (0.5,0.9)
D  (0.7,0.1) (0.7,0.3) (0.7,0.5) (0.7,0.7) (0.7,0.9)
E  (0.9,0.1) (0.9,0.3) (0.9,0.5) (0.9,0.7) (0.9,0.9)
```

### 座標の範囲

- **X座標**: 0.1 ～ 0.9 (0.2刻み)
- **Y座標**: 0.1 ～ 0.9 (0.2刻み)

### 座標計算式

位置文字列（例: "A1", "E5"）から座標への変換：

```python
row = ord(position[0]) - ord('A')  # 0-4
col = int(position[1]) - 1          # 0-4
x = col * 0.2 + 0.1                # 0.1, 0.3, 0.5, 0.7, 0.9
y = row * 0.2 + 0.1                # 0.1, 0.3, 0.5, 0.7, 0.9
```

### 主要位置

| 位置 | 座標 | 説明 |
|------|------|------|
| A1 | (0.1, 0.1) | 左上 |
| A5 | (0.1, 0.9) | 左下 |
| E1 | (0.9, 0.1) | 右上 |
| E5 | (0.9, 0.9) | 右下 |
| C3 | (0.5, 0.5) | 中央 |

---

## リクエスト形式

### 標準生成リクエスト

```json
{
  "input": "プロンプト",
  "model": "nai-diffusion-4-5-curated",
  "action": "generate",
  "parameters": {
    "width": 832,
    "height": 1216,
    "scale": 5.0,
    "sampler": "k_euler",
    "steps": 28,
    "seed": 0,
    "n_samples": 1,
    "ucPreset": 0,
    "qualityToggle": true,
    "sm": false,
    "sm_dyn": false,
    "dynamic_thresholding": false,
    "controlnet_strength": 1.0,
    "legacy": false,
    "add_original_image": false,
    "cfg_rescale": 0.0,
    "noise_schedule": "native",
    "legacy_v3_extend": false,
    "skip_cfg_above_sigma": null,
    "params_version": 1,
    "v4_prompt": { ... },
    "v4_negative_prompt": { ... }
  }
}
```

### 主要パラメータ

| パラメータ | 型 | 範囲 | デフォルト | 説明 |
|-----------|---|------|----------|------|
| width | int | 64-2048 | 832 | 画像幅（64刻み） |
| height | int | 64-2048 | 1216 | 画像高さ（64刻み） |
| scale | float | 0.0-10.0 | 5.0 | CFGスケール |
| steps | int | 1-50 | 28 | サンプリングステップ数 |
| seed | int | 0-9999999999 | 0 | ランダムシード |
| n_samples | int | 1-4 | 1 | 生成枚数 |
| noise_schedule | string | - | "native" | ノイズスケジュール |
| sm | boolean | - | false | SMEA有効化 |
| sm_dyn | boolean | - | false | SMEA Dynamic有効化 |

### noise_schedule オプション

- `native` (デフォルト)
- `karras`
- `exponential`
- `polyexponential`

---

## レスポンス形式

### 成功時

- **Content-Type**: `application/zip`
- **内容**: 生成された画像を含むZIPファイル
- ZIPファイル内に1枚以上のPNG/JPEGファイルが含まれる

### エラー時

- **Content-Type**: `application/json`
- **ステータスコード**: 400, 401, 500など

```json
{
  "statusCode": 500,
  "message": "Internal Server Error"
}
```

---

## 実装時の注意点

### 1. V4/V4.5モデルの判定

```python
is_v4_model = model.startswith("nai-diffusion-4")
```

### 2. SMEAサンプラーの変換

```python
if sampler == "nai_smea":
    actual_sampler = "k_euler_ancestral"
    sm = True
elif sampler == "nai_smea_dyn":
    actual_sampler = "k_euler_ancestral"
    sm = True
    sm_dyn = True
```

### 3. キャラクタープロンプトの構築

**重要なポイント:**
- キャラクターには必ず座標を指定
- `centers`フィールドは配列形式
- AUTO位置の場合は実装側で座標を割り当て
- ポジティブとネガティブで配列の長さを一致させる

```python
# 正しい構造
char_caption = {
    "char_caption": "1girl, blonde hair",
    "centers": [{"x": 0.1, "y": 0.1}]
}

# 間違った構造（500エラーの原因）
char_caption = {
    "char_caption": "1girl, blonde hair",
    "centers": []  # 空配列はNG
}
```

### 4. use_coords と use_order の設定

```python
# 座標を使用する場合
parameters["v4_prompt"] = {
    "use_coords": True,
    "use_order": False,
    ...
}

# 順序のみの場合（座標なし）
parameters["v4_prompt"] = {
    "use_coords": False,
    "use_order": True,
    ...
}
```

### 5. V3以前との互換性

```python
if is_v4_model:
    # V4/V4.5用の構造化プロンプト
    parameters["v4_prompt"] = { ... }
    parameters["v4_negative_prompt"] = { ... }
else:
    # V3以前の従来形式
    parameters["negative_prompt"] = negative_prompt
```

### 6. AUTO位置の自動割り当て

推奨される配置順序：
1. A1 (0.1, 0.1) - 左上
2. E5 (0.9, 0.9) - 右下
3. C3 (0.5, 0.5) - 中央
4. E1 (0.9, 0.1) - 右上
5. A5 (0.1, 0.9) - 左下

既に使用されている座標を避けて割り当てる必要があります。

### 7. よくあるエラー

| エラー | 原因 | 解決方法 |
|-------|------|---------|
| 500 Internal Server Error | キャラクタープロンプトの構造が不正 | 座標を正しく指定、centersを空配列にしない |
| 500 Internal Server Error | 座標が範囲外 | 0.1-0.9の範囲で指定 |
| 500 Internal Server Error | char_captionsの数が不一致 | v4_promptとv4_negative_promptで配列の長さを一致 |
| 401 Unauthorized | APIキーが無効 | 正しいAPIキーを設定 |

---

## 参考リンク

- [NovelAI公式サイト](https://novelai.net/)
- [NovelAI API ドキュメント](https://image.novelai.net/docs/)
- [NovelAI Python SDK](https://github.com/LlmKira/novelai-python)
- [ComfyUI_NAIDGenerator（参考実装）](https://github.com/bedovyy/ComfyUI_NAIDGenerator)

---

## 更新履歴

- 2025-01-XX: 初版作成
  - V4/V4.5モデルの仕様を調査
  - キャラクタープロンプトの構造を解明
  - 座標システムの正確な値を特定
  - 実装時の注意点をまとめ
