import os
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy
# OpenAI APIを使う場合は pip install openai が必要です
import google.generativeai as genai

genai.configure(api_key="API_KEY_HERE")

def generate_visualizations(file_path, output_dir):
    # 1. 音声の読み込み
    y, sr = librosa.load(file_path, duration=30)

    # 2. クロマグラムの生成
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # --- ここで画像保存処理（plt.savefig等）をこれまで通り実行 ---

    # 3. 数学的特徴の抽出（★ここで chroma を正しく渡す）
    props = analyze_music_properties(chroma) 
    
    # 4. Gemini API または ルールベースでコメント生成
    ai_text = get_ai_commentary_gemini(props) # Geminiを使う場合
    
    return {
        'chroma': 'plots/chroma.png',
        'ssm': 'plots/ssm.png',
        'ai_comment': ai_text
    }
    
def analyze_music_properties(chroma):
    """
    クロマデータから数学的特徴を抽出する
    """
    # 各音の出現頻度の平均
    mean_chroma = np.mean(chroma, axis=1)
    # 和音の複雑さ（エントロピー）: 値が高いほど多くの音が混ざり、複雑
    complexity = entropy(mean_chroma)
    # 音の強弱の激しさ（標準偏差）
    energy_std = np.std(chroma)
    
    return {
        "complexity": float(complexity),
        "energy_std": float(energy_std),
        "top_notes": np.argsort(mean_chroma)[-3:].tolist() # よく使われている音トップ3
    }

def get_ai_commentary_gemini(properties):
    """
    数値を元にGeminiに専門家としての解説を依頼する
    """
    model = genai.GenerativeModel('gemini-flash-latest') # 軽量で高速なモデル
    
    prompt = f"""
    あなたはクラシック音楽と数学に精通した分析家です。
    以下の音楽解析データ（30秒間）を元に、この曲の「数学的な秩序」について150文字程度で専門家らしく解説してください。
    
    【データ】
    - 和音の複雑性（エントロピー）: {properties['complexity']:.2f}
    - 音響エネルギーの分散（ダイナミクス）: {properties['energy_std']:.2f}
    - 主要な音（0=C, 1=C#...）: {properties['top_notes']}
    
    数値が低いほど秩序だったバロック風、高いほど色彩豊かな近代風として、
    音楽的なスタイルと数学的な特徴を関連づけて述べてください。
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI解析エラー: {str(e)}"
