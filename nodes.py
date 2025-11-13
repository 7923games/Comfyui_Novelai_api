"""
NovelAI API Node Implementation
"""

import requests
import io
import zipfile
import numpy as np
import torch
from PIL import Image
import json


class NovelAIImageGenerator:
    """
    NovelAI画像生成ノード
    NovelAI APIを使用して画像を生成します
    """

    def __init__(self):
        self.api_url = "https://image.novelai.net"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "prompt": ("STRING", {
                    "default": "1girl, masterpiece, best quality",
                    "multiline": True,
                }),
                "negative_prompt": ("STRING", {
                    "default": "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers",
                    "multiline": True,
                }),
                "model": ([
                    "nai-diffusion-4-5-full",
                    "nai-diffusion-4-5-curated",
                    "nai-diffusion-4-full",
                    "nai-diffusion-3",
                    "nai-diffusion-2",
                    "nai-diffusion",
                ], {
                    "default": "nai-diffusion-4-5-curated"
                }),
                "width": ("INT", {
                    "default": 832,
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                }),
                "height": ("INT", {
                    "default": 1216,
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                }),
                "steps": ("INT", {
                    "default": 28,
                    "min": 1,
                    "max": 50,
                }),
                "scale": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                }),
                "sampler": ([
                    "k_euler",
                    "k_euler_ancestral",
                    "k_dpm_2",
                    "k_dpm_2_ancestral",
                    "k_lms",
                    "k_dpmpp_2s_ancestral",
                    "k_dpmpp_sde",
                    "k_dpmpp_2m",
                    "k_dpmpp_2m_sde",
                    "k_dpmpp_3m_sde",
                    "k_dpm_adaptive",
                    "k_dpm_fast",
                    "plms",
                    "ddim",
                    "ddim_v3",
                    "nai_smea",
                    "nai_smea_dyn",
                ], {
                    "default": "k_euler"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999999999,
                }),
                "n_samples": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                }),
            },
            "optional": {
                "noise_schedule": ([
                    "native",
                    "karras",
                    "exponential",
                    "polyexponential",
                ], {
                    "default": "native"
                }),
                "character_prompts": ("NOVEL_AI_CHARACTER_LIST",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "NovelAI"

    def generate(self, api_key, prompt, negative_prompt, model, width, height,
                 steps, scale, sampler, seed, n_samples, noise_schedule="native",
                 character_prompts=None):
        """
        NovelAI APIを使用して画像を生成
        """

        if not api_key:
            raise ValueError("APIキーが設定されていません")

        # V4/V4.5モデルかどうかを判定
        is_v4_model = model.startswith("nai-diffusion-4")

        # SMEA系サンプラーの処理
        sm = False
        sm_dyn = False
        actual_sampler = sampler

        if sampler == "nai_smea":
            actual_sampler = "k_euler_ancestral"
            sm = True
        elif sampler == "nai_smea_dyn":
            actual_sampler = "k_euler_ancestral"
            sm = True
            sm_dyn = True

        # パラメータの構築
        parameters = {
            "width": width,
            "height": height,
            "scale": scale,
            "sampler": actual_sampler,
            "steps": steps,
            "seed": seed,
            "n_samples": n_samples,
            "ucPreset": 0,
            "qualityToggle": True,
            "sm": sm,
            "sm_dyn": sm_dyn,
            "dynamic_thresholding": False,
            "controlnet_strength": 1.0,
            "legacy": False,
            "add_original_image": False,
            "cfg_rescale": 0.0,
            "noise_schedule": noise_schedule,
            "legacy_v3_extend": False,
            "skip_cfg_above_sigma": None,
            "params_version": 1,
        }

        # V4/V4.5モデルの場合は構造化プロンプトを使用
        if is_v4_model:
            # キャラクターキャプションを構築
            char_captions = []
            has_coords = False
            if character_prompts is not None:
                for char in character_prompts:
                    char_caption = {
                        "char_caption": char["prompt"]
                    }
                    # centerが指定されている場合のみcentersを追加
                    if char["center"] is not None:
                        char_caption["centers"] = [char["center"]]
                        has_coords = True
                    # centerがNoneの場合はcentersフィールドを省略
                    char_captions.append(char_caption)

            parameters["v4_prompt"] = {
                "use_coords": has_coords,
                "use_order": not has_coords,  # coordsを使わない場合はorderを使う
                "caption": {
                    "base_caption": prompt,
                    "char_captions": char_captions
                }
            }

            # ネガティブプロンプトのchar_captionsも構築
            # ポジティブと同じ数のchar_captionsが必要
            negative_char_captions = []
            if character_prompts is not None:
                for char in character_prompts:
                    neg_char_caption = {
                        "char_caption": char["uc"] if char["uc"] else ""
                    }
                    # centerが指定されている場合のみcentersを追加
                    if char["center"] is not None:
                        neg_char_caption["centers"] = [char["center"]]
                    # centerがNoneの場合はcentersフィールドを省略
                    negative_char_captions.append(neg_char_caption)

            parameters["v4_negative_prompt"] = {
                "caption": {
                    "base_caption": negative_prompt,
                    "char_captions": negative_char_captions
                }
            }
        else:
            # V3以前のモデルでは従来のnegative_promptを使用
            parameters["negative_prompt"] = negative_prompt

        # リクエストボディの構築
        payload = {
            "input": prompt,
            "model": model,
            "action": "generate",
            "parameters": parameters
        }

        # ヘッダーの設定
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            # デバッグ: リクエスト内容をログ出力
            print("=== NovelAI API Request Debug ===")
            print(f"Model: {model}")
            if is_v4_model and character_prompts:
                print(f"Character Prompts Count: {len(character_prompts)}")
                print(f"v4_prompt: {json.dumps(parameters.get('v4_prompt'), indent=2, ensure_ascii=False)}")
                print(f"v4_negative_prompt: {json.dumps(parameters.get('v4_negative_prompt'), indent=2, ensure_ascii=False)}")
            print("=================================")

            # API リクエスト
            response = requests.post(
                f"{self.api_url}/ai/generate-image",
                headers=headers,
                json=payload,
                timeout=120
            )

            # レスポンスのステータスコードとエラー内容を確認
            if response.status_code != 200:
                print(f"Error Response Status: {response.status_code}")
                print(f"Error Response Body: {response.text}")

            response.raise_for_status()

            # ZIPファイルから画像を抽出
            images = []
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                for file_name in zip_file.namelist():
                    if file_name.endswith(('.png', '.jpg', '.jpeg')):
                        with zip_file.open(file_name) as image_file:
                            img = Image.open(image_file)
                            img = img.convert('RGB')
                            # PIL ImageをTensorに変換
                            img_array = np.array(img).astype(np.float32) / 255.0
                            img_tensor = torch.from_numpy(img_array)[None,]
                            images.append(img_tensor)

            if not images:
                raise ValueError("生成された画像が見つかりませんでした")

            # 複数画像をバッチとして結合
            output = torch.cat(images, dim=0)

            return (output,)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"NovelAI API リクエストエラー: {str(e)}")
        except zipfile.BadZipFile:
            raise RuntimeError("レスポンスがZIPファイルではありません")
        except Exception as e:
            raise RuntimeError(f"画像生成エラー: {str(e)}")


class NovelAIImageToImage:
    """
    NovelAI Image-to-Image ノード
    既存の画像を基に新しい画像を生成します
    """

    def __init__(self):
        self.api_url = "https://image.novelai.net"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "1girl, masterpiece, best quality",
                    "multiline": True,
                }),
                "negative_prompt": ("STRING", {
                    "default": "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers",
                    "multiline": True,
                }),
                "model": ([
                    "nai-diffusion-4-5-full",
                    "nai-diffusion-4-5-curated",
                    "nai-diffusion-4-full",
                    "nai-diffusion-3",
                    "nai-diffusion-2",
                    "nai-diffusion",
                ], {
                    "default": "nai-diffusion-4-5-curated"
                }),
                "strength": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "noise": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "steps": ("INT", {
                    "default": 28,
                    "min": 1,
                    "max": 50,
                }),
                "scale": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                }),
                "sampler": ([
                    "k_euler",
                    "k_euler_ancestral",
                    "k_dpm_2",
                    "k_dpm_2_ancestral",
                    "k_lms",
                    "k_dpmpp_2s_ancestral",
                    "k_dpmpp_sde",
                    "k_dpmpp_2m",
                    "k_dpmpp_2m_sde",
                    "k_dpmpp_3m_sde",
                    "k_dpm_adaptive",
                    "k_dpm_fast",
                    "plms",
                    "ddim",
                    "ddim_v3",
                    "nai_smea",
                    "nai_smea_dyn",
                ], {
                    "default": "k_euler"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999999999,
                }),
            },
            "optional": {
                "character_prompts": ("NOVEL_AI_CHARACTER_LIST",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "NovelAI"

    def generate(self, api_key, image, prompt, negative_prompt, model,
                 strength, noise, steps, scale, sampler, seed, character_prompts=None):
        """
        Image-to-Image生成
        """

        if not api_key:
            raise ValueError("APIキーが設定されていません")

        # 入力画像を処理（最初の画像のみ使用）
        input_image = image[0]
        img_array = (input_image.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_array)

        # 画像をバイト列に変換
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # V4/V4.5モデルかどうかを判定
        is_v4_model = model.startswith("nai-diffusion-4")

        # SMEA系サンプラーの処理
        sm = False
        sm_dyn = False
        actual_sampler = sampler

        if sampler == "nai_smea":
            actual_sampler = "k_euler_ancestral"
            sm = True
        elif sampler == "nai_smea_dyn":
            actual_sampler = "k_euler_ancestral"
            sm = True
            sm_dyn = True

        # パラメータの構築
        parameters = {
            "scale": scale,
            "sampler": actual_sampler,
            "steps": steps,
            "seed": seed,
            "n_samples": 1,
            "strength": strength,
            "noise": noise,
            "ucPreset": 0,
            "qualityToggle": True,
            "sm": sm,
            "sm_dyn": sm_dyn,
            "params_version": 1,
        }

        # V4/V4.5モデルの場合は構造化プロンプトを使用
        if is_v4_model:
            # キャラクターキャプションを構築
            char_captions = []
            has_coords = False
            if character_prompts is not None:
                for char in character_prompts:
                    char_caption = {
                        "char_caption": char["prompt"]
                    }
                    # centerが指定されている場合のみcentersを追加
                    if char["center"] is not None:
                        char_caption["centers"] = [char["center"]]
                        has_coords = True
                    # centerがNoneの場合はcentersフィールドを省略
                    char_captions.append(char_caption)

            parameters["v4_prompt"] = {
                "use_coords": has_coords,
                "use_order": not has_coords,  # coordsを使わない場合はorderを使う
                "caption": {
                    "base_caption": prompt,
                    "char_captions": char_captions
                }
            }

            # ネガティブプロンプトのchar_captionsも構築
            # ポジティブと同じ数のchar_captionsが必要
            negative_char_captions = []
            if character_prompts is not None:
                for char in character_prompts:
                    neg_char_caption = {
                        "char_caption": char["uc"] if char["uc"] else ""
                    }
                    # centerが指定されている場合のみcentersを追加
                    if char["center"] is not None:
                        neg_char_caption["centers"] = [char["center"]]
                    # centerがNoneの場合はcentersフィールドを省略
                    negative_char_captions.append(neg_char_caption)

            parameters["v4_negative_prompt"] = {
                "caption": {
                    "base_caption": negative_prompt,
                    "char_captions": negative_char_captions
                }
            }
        else:
            # V3以前のモデルでは従来のnegative_promptを使用
            parameters["negative_prompt"] = negative_prompt

        # マルチパートフォームデータの準備
        files = {
            'image': ('input.png', img_byte_arr, 'image/png'),
        }

        # ヘッダーの設定（Content-Typeは自動設定）
        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        # JSONデータをフォームデータに追加
        data = {
            'input': prompt,
            'model': model,
            'action': 'img2img',
            'parameters': json.dumps(parameters)
        }

        try:
            # デバッグ: リクエスト内容をログ出力
            print("=== NovelAI API Request Debug (img2img) ===")
            print(f"Model: {model}")
            if is_v4_model and character_prompts:
                print(f"Character Prompts Count: {len(character_prompts)}")
                print(f"v4_prompt: {json.dumps(parameters.get('v4_prompt'), indent=2, ensure_ascii=False)}")
                print(f"v4_negative_prompt: {json.dumps(parameters.get('v4_negative_prompt'), indent=2, ensure_ascii=False)}")
            print("===========================================")

            # API リクエスト
            response = requests.post(
                f"{self.api_url}/ai/generate-image",
                headers=headers,
                data=data,
                files=files,
                timeout=120
            )

            # レスポンスのステータスコードとエラー内容を確認
            if response.status_code != 200:
                print(f"Error Response Status: {response.status_code}")
                print(f"Error Response Body: {response.text}")

            response.raise_for_status()

            # ZIPファイルから画像を抽出
            images = []
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                for file_name in zip_file.namelist():
                    if file_name.endswith(('.png', '.jpg', '.jpeg')):
                        with zip_file.open(file_name) as image_file:
                            img = Image.open(image_file)
                            img = img.convert('RGB')
                            img_array = np.array(img).astype(np.float32) / 255.0
                            img_tensor = torch.from_numpy(img_array)[None,]
                            images.append(img_tensor)

            if not images:
                raise ValueError("生成された画像が見つかりませんでした")

            output = torch.cat(images, dim=0)

            return (output,)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"NovelAI API リクエストエラー: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"画像生成エラー: {str(e)}")


class NovelAICharacterPrompt:
    """
    NovelAIキャラクタープロンプトノード
    V4/V4.5モデルで使用する個別のキャラクター定義
    """

    @classmethod
    def INPUT_TYPES(cls):
        # 位置マップ（5x5グリッド）
        positions = ["AUTO"]
        for row in ["A", "B", "C", "D", "E"]:
            for col in range(1, 6):
                positions.append(f"{row}{col}")

        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "1girl",
                    "multiline": True,
                }),
                "position": (positions, {
                    "default": "AUTO"
                }),
            },
            "optional": {
                "character_negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                }),
            }
        }

    RETURN_TYPES = ("NOVEL_AI_CHARACTER",)
    FUNCTION = "create_character"
    CATEGORY = "NovelAI"

    def create_character(self, prompt, position, character_negative_prompt="", enabled=True):
        """
        キャラクタープロンプトを作成
        """
        # 位置を座標に変換
        if position == "AUTO":
            center = None
        else:
            # A1-E5形式を座標に変換
            row = ord(position[0]) - ord('A')  # 0-4
            col = int(position[1]) - 1  # 0-4
            # 正規化された座標（0.1-0.9の範囲、0.2刻み）
            x = col * 0.2 + 0.1
            y = row * 0.2 + 0.1
            center = {"x": x, "y": y}

        character = {
            "prompt": prompt,
            "uc": character_negative_prompt,
            "center": center,
            "enabled": enabled
        }

        return (character,)


class NovelAICharacterPromptCombine:
    """
    複数のキャラクタープロンプトを結合するノード
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "character_1": ("NOVEL_AI_CHARACTER",),
                "character_2": ("NOVEL_AI_CHARACTER",),
                "character_3": ("NOVEL_AI_CHARACTER",),
                "character_4": ("NOVEL_AI_CHARACTER",),
                "character_5": ("NOVEL_AI_CHARACTER",),
            }
        }

    RETURN_TYPES = ("NOVEL_AI_CHARACTER_LIST",)
    FUNCTION = "combine"
    CATEGORY = "NovelAI"

    def combine(self, character_1=None, character_2=None, character_3=None,
                character_4=None, character_5=None):
        """
        キャラクタープロンプトを結合
        AUTO位置のキャラクターには自動的に位置を割り当て
        """
        characters = []
        for char in [character_1, character_2, character_3, character_4, character_5]:
            if char is not None:
                characters.append(char)

        print(f"\n=== Character Prompt Combine Debug ===")
        print(f"Total characters: {len(characters)}")

        # AUTO位置のキャラクターに自動的に位置を割り当て
        # 利用可能な位置リスト（見栄えの良い配置順）
        available_positions = [
            {"x": 0.1, "y": 0.1},  # A1 - 左上
            {"x": 0.9, "y": 0.9},  # E5 - 右下
            {"x": 0.5, "y": 0.5},  # C3 - 中央
            {"x": 0.9, "y": 0.1},  # E1 - 右上
            {"x": 0.1, "y": 0.9},  # A5 - 左下
        ]

        # 既に使用されている位置を収集
        used_positions = set()
        for i, char in enumerate(characters):
            if char["center"] is not None:
                pos_key = f"{char['center']['x']},{char['center']['y']}"
                used_positions.add(pos_key)
                print(f"Character {i+1}: Pre-assigned position {pos_key}")

        # AUTO位置のキャラクターに位置を割り当て
        auto_position_index = 0
        for i, char in enumerate(characters):
            if char["center"] is None:
                print(f"Character {i+1}: AUTO position, assigning...")
                # 利用可能な位置を探す
                while auto_position_index < len(available_positions):
                    pos = available_positions[auto_position_index]
                    pos_key = f"{pos['x']},{pos['y']}"
                    auto_position_index += 1

                    if pos_key not in used_positions:
                        char["center"] = pos.copy()  # コピーして割り当て
                        used_positions.add(pos_key)
                        print(f"  → Assigned to {pos_key}")
                        break
            else:
                print(f"Character {i+1}: Position already set to {char['center']['x']},{char['center']['y']}")

        print(f"=== End Character Prompt Combine Debug ===\n")
        return (characters,)


# ノードの登録
NODE_CLASS_MAPPINGS = {
    "NovelAIImageGenerator": NovelAIImageGenerator,
    "NovelAIImageToImage": NovelAIImageToImage,
    "NovelAICharacterPrompt": NovelAICharacterPrompt,
    "NovelAICharacterPromptCombine": NovelAICharacterPromptCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NovelAIImageGenerator": "NovelAI Image Generator",
    "NovelAIImageToImage": "NovelAI Image to Image",
    "NovelAICharacterPrompt": "NovelAI Character Prompt",
    "NovelAICharacterPromptCombine": "NovelAI Character Prompt Combine",
}
