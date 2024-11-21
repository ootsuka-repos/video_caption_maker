import os
import math
import cv2
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# モデルの読み込みをグローバルに行う
def load_blip2_model(model_type="coco"):
    if model_type == "coco":
        model_name = "Salesforce/blip2-opt-2.7b-coco"
    elif model_type == "pretrain":
        model_name = "Salesforce/blip2-opt-2.7b"
    else:
        raise ValueError(f"サポートされていないモデルタイプ: {model_type}")
    
    print(f"モデル{model_name}を読み込んでいます...")
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("モデルの読み込みが完了しました。")
    return processor, model

# モデルのアンロード
def unload_blip2_model(processor, model):
    del model
    del processor
    torch.cuda.empty_cache()

def generate_video_captions(
    video_path: str, 
    processor, 
    model, 
    output_dir: str = None, 
    frame_interval_percentage: float = 0.05,
    num_beams: int = 3,
    use_nucleus_sampling: bool = False,
    max_length: int = 30,
    min_length: int = 10,
    top_p: float = 0.9
):
    """
    動画からフレームを抽出し、キャプションを生成する関数
    """
    # 出力ディレクトリ設定
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(video_path), 
            f"{os.path.splitext(os.path.basename(video_path))[0]}_captions"
        )
    
    # キャプション保存用ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # ビデオキャプチャーオブジェクトの作成
    cap = cv2.VideoCapture(video_path)
    
    # フレームカウンター
    frame_count = 0
    saved_frame_count = 0
    
    # 動画の詳細情報
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    # フレーム間隔を動画の総フレーム数に基づいて計算
    frame_interval = math.ceil(total_frames * frame_interval_percentage)
    
    print(f"動画情報: {total_frames}フレーム, {fps:.2f} FPS, 長さ: {duration:.2f}秒")
    print(f"フレーム間隔: {frame_interval}フレーム")
    
    # テキストファイルに動画メタデータを保存
    meta_file = os.path.join(output_dir, 'video_metadata.txt')
    with open(meta_file, 'w', encoding='utf-8') as f:
        f.write(f"ファイル: {os.path.basename(video_path)}\n")
        f.write(f"総フレーム数: {total_frames}\n")
        f.write(f"フレームレート: {fps:.2f} FPS\n")
        f.write(f"動画の長さ: {duration:.2f}秒\n")
        f.write(f"フレーム間隔: {frame_interval}フレーム\n")
    
    # キャプションを1つのファイルに保存
    caption_file_path = os.path.join(output_dir, 'captions.txt')
    
    try:
        with open(caption_file_path, 'w', encoding='utf-8') as caption_file:
            while True:
                # フレームの読み込み
                ret, frame = cap.read()
                
                # フレームがない場合は終了
                if not ret:
                    break
                
                # 指定されたインターバルでフレーム抽出
                if frame_count % frame_interval == 0:
                    # OpenCVの画像をPIL Imageに変換
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # キャプション生成
                    inputs = processor(images=pil_image, return_tensors="pt").to(model.device)
                    generated_ids = model.generate(
                        **inputs, 
                        num_beams=num_beams,
                        max_length=max_length,
                        min_length=min_length,
                        top_p=top_p if use_nucleus_sampling else None,
                        do_sample=use_nucleus_sampling,
                        repetition_penalty=1.0
                    )
                    captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    
                    # キャプションをファイルに追記
                    caption_text = f"{captions[0]}"
                    caption_file.write(caption_text)
                    
                    print(f"フレーム {saved_frame_count} のキャプション: {captions[0]}")
                    
                    saved_frame_count += 1
                
                frame_count += 1
        
        print(f"総キャプション生成数: {saved_frame_count}")
        return saved_frame_count
    
    finally:
        # ビデオキャプチャーを必ず解放
        cap.release()

def process_all_videos_in_folder(
    folder_path: str,
    output_base_dir: str,
    frame_interval_percentage: float = 0.05,
    model_type: str = "coco",
    num_beams: int = 3,
    use_nucleus_sampling: bool = False,
    max_length: int = 30,
    min_length: int = 10,
    top_p: float = 0.9
):
    """
    指定されたフォルダ内のすべての動画を処理してキャプションを生成する関数
    """
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print("指定されたフォルダに動画ファイルが見つかりませんでした。")
        return
    
    print(f"フォルダ内の動画ファイル: {video_files}")
    
    # モデルの読み込み
    processor, model = load_blip2_model(model_type)
    
    try:
        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            output_dir = os.path.join(output_base_dir, os.path.splitext(video_file)[0])
            
            print(f"処理中: {video_file}")
            generate_video_captions(
                video_path=video_path,
                processor=processor,
                model=model,
                output_dir=output_dir,
                frame_interval_percentage=frame_interval_percentage,
                num_beams=num_beams,
                use_nucleus_sampling=use_nucleus_sampling,
                max_length=max_length,
                min_length=min_length,
                top_p=top_p
            )
    finally:
        # モデルのアンロード
        unload_blip2_model(processor, model)

# 使用例
if __name__ == "__main__":
    folder_path = r"C:\Users\user\Desktop\git\dataset\outputs\480x720\videos"
    output_base_dir = r"C:\Users\user\Desktop\git\dataset\outputs\captions"
    
    process_all_videos_in_folder(
        folder_path=folder_path,
        output_base_dir=output_base_dir,
        frame_interval_percentage=0.025,
        model_type="coco",
        num_beams=3,
        use_nucleus_sampling=False,
        max_length=100,
        min_length=10,
        top_p=0.9
    )