import pandas as pd
import os
import shutil
import cv2
from mtcnn_cv2 import MTCNN
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import re
import requests
import json


class Classifier:

    def __init__(self, image_path, run_name) -> None:
        self.image_path = image_path
        if run_name is None:
            self.run_date = datetime.utcnow().strftime("%Y%m%d_%H%M")
        else:
            self.run_date = run_name

    def download_images(self, url_list, read_from_json = False, json_path = None):
        self.url_mapper = {}
        if self.image_path not in os.listdir():
            os.mkdir(f'./{self.image_path}')
        if read_from_json:
            with open(f'{json_path}') as f:
                content = f.read()
            url_dict = json.loads(content)
            url_list = url_dict['url_list']
        for n,url in enumerate(url_list):
            self.url_mapper[f'{n}.jpg'] = url
            img_data = requests.get(url).content
            with open(f'./{self.image_path}/{n}.jpg', 'wb') as handler:
                handler.write(img_data)
    
    def delete_dirs(self):
        shutil.rmtree(f'./{self.image_path}')
        shutil.rmtree(f'./prediction_files/single_faces/{self.run_date}')
        shutil.rmtree(f'./prediction_files/single_faces_cropped/{self.run_date}')

    
    def crop_images(self):
        images = os.listdir(f'./{self.image_path}')
        detector = MTCNN()
        self.cropped_dict = {}
        for image in images:
            img = cv2.imread(f'./{self.image_path}/{image}')
            colored = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = detector.detect_faces(colored)
            self.cropped_dict[image] = []
            if len(result) == 0:
                faces_new = []
            elif len(result) > 1:
                faces_new = [a['box']
                             for a in result if a['confidence'] >= 0.99]
            else:
                faces_new = [a['box'] for a in result]

            if len(faces_new) == 0:
                pass
            else:
                for face in faces_new:
                    x, y, w, h = face
                    face_cropped = img[y:y + h, x:x + w]
                    self.cropped_dict[image].append(face_cropped)

    def clean_generate_dirs(self):

        if 'prediction_files' not in os.listdir():
            os.mkdir('./prediction_files')

            # Prediction folder for no faces
            # os.mkdir('./prediction_files/no_faces')
            # os.mkdir('./prediction_files/no_faces/test')

            # Prediction folder for multiple faces
            # os.mkdir('./prediction_files/multiple_faces')
            # os.mkdir('./prediction_files/multiple_faces/original')
            # os.mkdir('./prediction_files/multiple_faces/test')

            # Prediction folder for gender prediction
            os.mkdir('./prediction_files/single_faces')
            # os.mkdir('./prediction_files/single_faces/test')

            # Prediction folder for age prediction
            os.mkdir('./prediction_files/single_faces_cropped')
            # os.mkdir('./prediction_files/single_faces_cropped/test')

        # os.mkdir(f'./prediction_files/no_faces/{self.run_date}')
        # os.mkdir(f'./prediction_files/no_faces/{self.run_date}/test')

        # os.mkdir(f'./prediction_files/multiple_faces/{self.run_date}')
        # os.mkdir(f'./prediction_files/multiple_faces/{self.run_date}/original')
        # os.mkdir(f'./prediction_files/multiple_faces/{self.run_date}/cropped')
        # os.mkdir(
        #     f'./prediction_files/multiple_faces/{self.run_date}/cropped/test')

        os.mkdir(f'./prediction_files/single_faces/{self.run_date}')
        os.mkdir(f'./prediction_files/single_faces/{self.run_date}/test')

        os.mkdir(f'./prediction_files/single_faces_cropped/{self.run_date}')
        os.mkdir(
            f'./prediction_files/single_faces_cropped/{self.run_date}/test')

    def move_write_images(self):
        for key in self.cropped_dict:
            if len(self.cropped_dict[key]) == 1:
                shutil.move(f'./{self.image_path}/{key}',
                            f'./prediction_files/single_faces/{self.run_date}/test/{key}')
                cv2.imwrite(
                    f'./prediction_files/single_faces_cropped/{self.run_date}/test/{key}', self.cropped_dict[key][0])
            # elif len(self.cropped_dict[key]) > 1:
            #     shutil.move(f'./{self.image_path}/{key}',
            #                 f'./prediction_files/multiple_faces/{self.run_date}/original/{key}')
            #     for n, img in enumerate(self.cropped_dict[key]):
            #         cv2.imwrite(
            #             f'./prediction_files/multiple_faces/{self.run_date}/cropped/test/{n}_{key}', img)
            # else:
            #     shutil.move(f'./{self.image_path}/{key}',
            #                 f'./prediction_files/no_faces/{self.run_date}/test/{key}')
            else:
                pass

    def load_models_generators(self, gender_models, age_models):
        self.model_genders = [keras.models.load_model(
            f'./models/{gender_model}') for gender_model in gender_models]
        self.model_ages = [keras.models.load_model(
            f'./models/{age_model}') for age_model in age_models]

        datagen_age = ImageDataGenerator(rescale=1./255)

        size_age = 224
        size = 249
        datagen_gender = ImageDataGenerator(rescale=1./255)
        datagen_age = ImageDataGenerator(rescale=1./255)

        self.test_gen_gender = datagen_gender.flow_from_directory(
            f'./prediction_files/single_faces/{self.run_date}',
            target_size=(size, size),
            batch_size=1,
            class_mode=None,
            shuffle=False
        )

        self.test_gen_age = datagen_age.flow_from_directory(
            f'./prediction_files/single_faces_cropped/{self.run_date}',
            target_size=(size_age, size_age),
            batch_size=1,
            class_mode=None,
            shuffle=False
        )

    def generate_output_files(self, output_file_extension=None, generate_excel=True, return_keys = None):
        gender_mapper = {0: 'W', 1: 'M'}
        age_mapper = {0: '0_17', 1: '18_34', 2: '35_59', 3: '60_inf'}

        gender_preds = np.zeros((len(self.test_gen_gender), 1))
        for model in self.model_genders:
            gender_preds += model.predict(self.test_gen_gender)
        gender_preds /= len(self.model_genders)
        gender_preds_class = np.round(gender_preds)

        age_preds = np.zeros((len(self.test_gen_age), len(age_mapper)))
        for model in self.model_ages:
            age_preds += model.predict(self.test_gen_age)

        age_preds /= len(self.model_ages)
        age_preds_class = np.argmax(age_preds, axis=1)

        file_to_preds = {
            re.sub('test/', '', self.test_gen_age.filenames[n]):
            {
                'gender_pred': gender_preds_class[n][0],
                'gender_pred_label': gender_mapper[gender_preds_class[n][0]],
                'age_pred': age_preds_class[n],
                'age_pred_label': age_mapper[age_preds_class[n]]
            } for n in range(len(gender_preds_class))
        }

        
        if generate_excel:
            pred_df = pd.DataFrame(file_to_preds).T
            if output_file_extension is None:
                pred_df.to_excel(f'file_to_preds_{self.run_date}.xlsx')
            else:
                pred_df.to_excel(f'file_to_preds_{output_file_extension}.xlsx')

        if return_keys is None:
            return_dict = file_to_preds
        else:
            return_dict = {key: {key_2: file_to_preds[key][key_2] for key_2 in return_keys} for key in file_to_preds}
        
        return return_dict
