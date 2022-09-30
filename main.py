import os
import shutil
import time
import streamlit as st
from PIL import Image


from stqdm import stqdm

import shutil
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import matplotlib.pyplot as plt
import re
import os
from IPython.display import Image

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras import Model, layers
from annoy import AnnoyIndex
import glob
from keras.applications.imagenet_utils import preprocess_input

from PIL import Image
import matplotlib.pyplot as plt
import japanize_matplotlib


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def get_html(url):
    res=requests.get(url)
    return res

def pages(url):
    res=get_html(url)
    soup_n = bs(res.content,"html.parser")
    nums = soup_n.find_all(class_="paging")[0]
    nums = nums.find_all("a")[-1].get('href')
    nums = nums.split("_")[-1].replace('/', '')
    page = int(nums)

    return page

def details(item):
    item = item.find_all(class_="product_img js-ecommerce-action")
    return {
        "name": item[0].get("syo_name"),
        "pic": item[0].img.get('src'),
        "URL":"https://www.buyma.com/item/"+str(item[0].get("syo_id"))+"/"
    } 

def df_maker(items_list):
    li = []
    
    for item in items_list:
        try:
            li.append(details(item))
        
        except:
            del item
    
    df = pd.DataFrame(li)
    return df

def scraper(url):
    items_list=[]
    n = pages(url)
    
    urls_list=[]
    
    for i in stqdm(range(1,n+1)):
        if i==1:
            urls_list.append(url)
        
        else:
            url=url.lstrip('/')
            url= url+"_"+str(i)+"/"
            urls_list.append(url)

    urls_list = urls_list[:15]
    
    items_list=[]
    for url in stqdm(urls_list):
        res=get_html(url)
        soup = bs(res.content,"html.parser")
        items = soup.find_all(class_= "product_lists")[0]
        items = items.find_all(class_="product js-psnlz-item")
        items_list+=[item for item in items]
    
    return items_list

def download_image(url, file_path):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(r.content)

def scr(title):
    
    url = "https://www.buyma.com/r/-C1002/"+title+"/"
    base_dir = "./"
    thumb_dir = os.path.join(base_dir, "pics_file")
    
    item_url=scraper(url)
    df = df_maker(item_url)
    
    base_model = VGG16(weights="imagenet")
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)

    for i in stqdm(range(len(df))):
        url=df["pic"][i]
        file_name = "{}.jpg".format(i)
        image_path = os.path.join(thumb_dir, file_name)
        download_image(url=url, file_path=image_path)

    if os.path.exists("./pics_file/target 2.jpg"):
        os.remove("./pics_file/target 2.jpg")
    
    
    dim = 4096
    annoy_model = AnnoyIndex(dim)
    numimg= 0
    glob_dir = os.path.join(thumb_dir, "*.jpg")
    
    
    files = glob.glob(glob_dir)
    files = sorted(files, key=natural_keys)
    
    for file in stqdm(files):
        img_path = file
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        fc2_features = model.predict(x,verbose=0)
        annoy_model.add_item(numimg, fc2_features[0])
        numimg += 1
    
    
    annoy_model.build(numimg)
    save_path = os.path.join(base_dir, "result.ann")
    annoy_model.save(save_path)
    annoy_model.unload()
    trained_model = AnnoyIndex(4096)
    trained_model.load("./result.ann")
    items = trained_model.get_nns_by_item(len(df), 21, search_k=-1, include_distances=False)

    return df, items

def main():
    st.title('buyma画像検索アプリ')
    
    #保存先の作成
    dir = './pics_file'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

    #アップロードされた画像を保存先へ移動
    uploaded_file=st.file_uploader("画像をアップロード")

    Image.open(uploaded_file).convert('RGB').save('target.jpg')
    new_path = shutil.move('./target.jpg', './pics_file')

    #検索名の入力
    with st.form('text_form'):
            search_text = st.text_input('商品名を入力')
            button = st.form_submit_button('Search Image')

    #スクレピングと類似画像検索の実行
    df, items = scr(search_text)

    #パス
    base_dir = "./"
    thumb_dir = os.path.join(base_dir, "pics_file")

    #画像の表示
    glob_target = os.path.join(thumb_dir, "target.jpg")
    image_target = Image.open(glob_target )
    st.image(image_target, caption='対象画像',width = 128)


    for i in range(20):
        glob_n = os.path.join(thumb_dir, f"{int(items[i+1])}.jpg")
        img_n = Image.open(glob_n)
        st.image(img_n, caption=f'レコメンド商品NO{i+1}',width = 128)
        st.caption(df["URL"][int(items[i+1])])



if __name__ == '__main__':
    main()