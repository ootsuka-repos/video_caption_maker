import os
import math
import cv2
import danbooru  # Importing the danbooru module as requested

def generate_video_captions(
    video_path: str, 
    output_dir: str = None, 
    frame_interval_percentage: float = 0.05
):
    """
    動画からフレームを抽出し、キャプションとタグを生成する関数
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
    
    # キャプションとタグを1つのファイルに保存
    caption_file_path = os.path.join(output_dir, 'captions.txt')
    
    try:
        with open(caption_file_path, 'w', encoding='utf-8') as output_file:
            while True:
                # フレームの読み込み
                ret, frame = cap.read()
                
                # フレームがない場合は終了
                if not ret:
                    break
                
                # 指定されたインターバルでフレーム抽出
                if frame_count % frame_interval == 0:
                    # 一時的な一意のフレーム画像を保存
                    temp_frame_path = os.path.join(output_dir, f"frame_{saved_frame_count}.jpg")
                    cv2.imwrite(temp_frame_path, frame)
                    
                    # タグ予測を実行
                    results = danbooru.predict_image_tags(temp_frame_path)
                    
                    # 結果をファイルに書き込み
                    output_file.write(f"{results['tags']}\n")
                    
                    print(f"フレーム {saved_frame_count} のタグ: {results['tags']}")
                    
                    # 一時ファイルを削除
                    os.remove(temp_frame_path)
                    
                    saved_frame_count += 1
                
                frame_count += 1
        
        print(f"総フレーム数: {saved_frame_count}")
        return saved_frame_count
    
    finally:
        # ビデオキャプチャーを必ず解放
        cap.release()

def process_all_videos_in_folder(
    folder_path: str,
    output_base_dir: str,
    frame_interval_percentage: float = 0.05
):
    """
    指定されたフォルダ内のすべての動画を処理してタグを生成する関数
    """
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print("指定されたフォルダに動画ファイルが見つかりませんでした。")
        return
    
    print(f"フォルダ内の動画ファイル: {video_files}")
    
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        output_dir = os.path.join(output_base_dir, os.path.splitext(video_file)[0])
        
        print(f"処理中: {video_file}")
        generate_video_captions(
            video_path=video_path,
            output_dir=output_dir,
            frame_interval_percentage=frame_interval_percentage
        )

# 使用例
if __name__ == "__main__":
    folder_path = r"C:\Users\user\Desktop\git\video_dataset_trim\outputs\720x480\videos"
    output_base_dir = r"C:\Users\user\Desktop\git\video_dataset_trim\outputs\captions"
    
    process_all_videos_in_folder(
        folder_path=folder_path,
        output_base_dir=output_base_dir,
        frame_interval_percentage=0.05
    )