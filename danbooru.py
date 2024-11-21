import numpy as np 
import onnxruntime as rt 
import pandas as pd 
from PIL import Image 
import huggingface_hub 
import functools

# グローバル変数としてモデルをキャッシュ
GLOBAL_MODEL = None
GLOBAL_TAGS_DF = None

MODEL_REPO = "SmilingWolf/wd-eva02-large-tagger-v3" 
MODEL_FILENAME = "model.onnx" 
LABEL_FILENAME = "selected_tags.csv" 

def load_model(model_repo=MODEL_REPO):
    """
    モデルを読み込み、キャッシュする関数
    
    Args:
        model_repo (str): モデルのリポジトリパス
    
    Returns:
        tuple: モデルセッション、タグデータフレーム、タグ情報
    """
    global GLOBAL_MODEL, GLOBAL_TAGS_DF
    
    # モデルがまだキャッシュされていない場合のみ読み込む
    if GLOBAL_MODEL is None or GLOBAL_TAGS_DF is None:
        # モデルとラベルをHugging Faceからダウンロード 
        csv_path = huggingface_hub.hf_hub_download(model_repo, LABEL_FILENAME) 
        model_path = huggingface_hub.hf_hub_download(model_repo, MODEL_FILENAME) 
    
        # タグ情報を読み込む 
        GLOBAL_TAGS_DF = pd.read_csv(csv_path) 
        tag_names, rating_indexes, general_indexes, character_indexes = load_labels(GLOBAL_TAGS_DF) 
    
        # モデルを読み込む 
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] 
        GLOBAL_MODEL = rt.InferenceSession(model_path, providers=providers)
    
    return GLOBAL_MODEL, GLOBAL_TAGS_DF

def load_labels(dataframe): 
    """ 
    データフレームからラベル情報を読み込む関数 
     
    Args: 
        dataframe (pandas.DataFrame): タグ情報が含まれるデータフレーム 
     
    Returns: 
        tuple: タグ名のリストと各カテゴリのインデックスリスト 
    """ 
    # タグ名から'_'を削除して読みやすくする 
    name_series = dataframe["name"].map(lambda x: x.replace("_", " ")) 
    tag_names = name_series.tolist() 
 
    # カテゴリごとのインデックスを取得 
    # カテゴリ9: 評価タグ 
    # カテゴリ0: 一般タグ 
    # カテゴリ4: キャラクタータグ 
    rating_indexes = list(np.where(dataframe["category"] == 9)[0]) 
    general_indexes = list(np.where(dataframe["category"] == 0)[0]) 
    character_indexes = list(np.where(dataframe["category"] == 4)[0]) 
     
    return tag_names, rating_indexes, general_indexes, character_indexes 
 
def predict_image_tags(image_path, model_repo=MODEL_REPO, general_thresh=0.55, character_thresh=0.85): 
    """ 
    画像からタグを予測する関数 
     
    Args: 
        image_path (str): 画像ファイルのパス 
        model_repo (str): 使用するモデルのリポジトリ 
        general_thresh (float): 一般タグの閾値 (デフォルト: 0.55) 
        character_thresh (float): キャラクタータグの閾値 (デフォルト: 0.85) 
     
    Returns: 
        dict: タグ情報を含む辞書 
    """ 
    # モデルとタグ情報を読み込む
    model, tags_df = load_model(model_repo)
    
    # タグ情報を読み込む 
    tag_names, rating_indexes, general_indexes, character_indexes = load_labels(tags_df) 
 
    # モデルの入力サイズを取得 
    _, height, width, _ = model.get_inputs()[0].shape 
 
    # 画像を準備 
    image = Image.open(image_path) 
     
    # RGBモードでない場合は変換 
    if image.mode != 'RGB': 
        image = image.convert('RGB') 
 
    # 画像を正方形にパディングとリサイズ 
    image_shape = image.size 
    max_dim = max(image_shape) 
    pad_left = (max_dim - image_shape[0]) // 2 
    pad_top = (max_dim - image_shape[1]) // 2 
 
    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255)) 
    padded_image.paste(image, (pad_left, pad_top)) 
    padded_image = padded_image.resize((height, width), Image.BICUBIC) 
 
    # NumPy配列に変換 
    image_array = np.asarray(padded_image, dtype=np.float32) 
    image_array = image_array[:, :, ::-1]  # RGBからBGRに変換 
    image_array = np.expand_dims(image_array, axis=0) 
 
    # タグ予測 
    input_name = model.get_inputs()[0].name 
    label_name = model.get_outputs()[0].name 
    preds = model.run([label_name], {input_name: image_array})[0] 
 
    # タグと予測確率を紐付け 
    labels = list(zip(tag_names, preds[0].astype(float))) 
 
    # 評価タグの処理 
    ratings_names = [labels[i] for i in rating_indexes] 
    rating = dict(ratings_names) 
 
    # 一般タグの処理 
    # 指定された閾値以上の確率を持つタグのみを選択 
    general_names = [labels[i] for i in general_indexes] 
    general_res = [x for x in general_names if x[1] > general_thresh] 
    general_res = dict(general_res) 
 
    # キャラクタータグの処理 
    # 指定された閾値以上の確率を持つタグのみを選択 
    character_names = [labels[i] for i in character_indexes] 
    character_res = [x for x in character_names if x[1] > character_thresh] 
    character_res = dict(character_res) 
 
    # 一般タグを確率の高い順にソートして文字列化 
    sorted_general_strings = sorted( 
        general_res.items(), 
        key=lambda x: x[1], 
        reverse=True, 
    ) 
    sorted_general_strings = [x[0] for x in sorted_general_strings] 
    sorted_general_strings = ", ".join(sorted_general_strings) 
 
    # 結果を辞書形式で返す 
    return { 
        'tags': sorted_general_strings,  # 確率の高い一般タグをカンマ区切りの文字列で 
        'rating': rating,  # 評価タグ（例：セーフ、R-18など） 
        'characters': character_res,  # キャラクタータグと確率 
        'general_tags': general_res  # 一般タグと確率の辞書 
    }