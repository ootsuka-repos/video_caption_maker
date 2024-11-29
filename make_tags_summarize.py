import os

# ベースディレクトリ
base_dir = "C:\\Users\\user\\Desktop\\git\\video_dataset_trim\\outputs\\captions"

# 出力ファイル名
output_file_tags = os.path.join(base_dir, "combined_selected_tags.txt")
output_file_videos = os.path.join(base_dir, "videos.txt")

# 出力ファイルを開く
with open(output_file_tags, "w", encoding="utf-8") as outfile_tags, open(output_file_videos, "w", encoding="utf-8") as outfile_videos:
    # 連番順にファイルを処理
    for i in range(1, 9999):  # 1から9998までの連番
        file_path = os.path.join(base_dir, str(i), "selected_tags.txt")
        
        # ファイルが存在するか確認
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as infile:
                # ファイルの内容を読み込む
                tags = infile.read().strip()
                # タグと参照txtのパスを出力ファイルに書き込む
                outfile_tags.write(f"{tags}\n")
                # videos.txtに対応するビデオパスを書き込む
                outfile_videos.write(f"videos/{i}.mp4\n")
        else:
            # ファイルが見つからない場合はスキップ
            continue

print(f"すべてのselected_tags.txtが {output_file_tags} に結合されました。")
print(f"対応するビデオパスが {output_file_videos} に書き込まれました。")