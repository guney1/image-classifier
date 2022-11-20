from classifier import *
import argparse
from unicodedata import name
import time


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-id', '--image_directory', type=str,
                        default='test_folder',
                        help='name of the dir, original images stored')

    parser.add_argument('-rl', '--read_from_local', type=str,
                        default='False', help='If the images ready or will be downloaded from urls')

    parser.add_argument('-url', '--url_list', nargs='+',
                        default=[],
                        help='Image urls to be downloaded')
    
    parser.add_argument('-jp', '--json_path', type=str,
                        default=None, help='Path to the json file of urls')

    parser.add_argument('-i', '--run_id', type=str,
                        default=None,
                        help='identifier for the run id, to generate prediction files, if None; %Y%m%d_%H%M')

    parser.add_argument('-p', '--process_data', type=str,
                        default='True',
                        help='If images already processed and inside prediction files, provide the corresponding run_id and set it to false')

    parser.add_argument('-mgs', '--model_gender_names', nargs='+',
                        default=['xception_v2_01_0.946.h5'],
                        help='name of the corresponding gender model names inside models dir')

    parser.add_argument('-mas', '--model_age_names', nargs='+',
                        default=['vgg_age_model_wiki_17_0.798.h5',
                                 'vgg_age_model_wiki_imdb_22_0.769.h5'],
                        help='name of the corresponding age model names inside models dir')

    parser.add_argument('-oe', '--output_file_extension', type=str,
                        default=datetime.utcnow().strftime("%Y%m%d_%H%M"),
                        help='name of the output file')

    args = parser.parse_args()

    model = Classifier(image_path=args.image_directory, run_name=args.run_id)

    gender_models = args.model_gender_names
    age_models = args.model_age_names

    if args.read_from_local == 'False':
        if args.json_path is not None:
            read_from_json = True
        else:
            read_from_json = False
        
        model.download_images(url_list=args.url_list, read_from_json=read_from_json, json_path=args.json_path)

    if args.process_data == 'True':
        model.clean_generate_dirs()
        model.crop_images()
        model.move_write_images()
    
    if len(os.listdir(f'./prediction_files/single_faces/{model.run_date}/test')) == 0:
        return {}
    else:
        model.load_models_generators(
            gender_models=gender_models, age_models=age_models)
        pred_dict = model.generate_output_files(
            output_file_extension=args.output_file_extension, generate_excel=False, return_keys=['gender_pred_label', 'age_pred_label'])
        
        model.delete_dirs()
        
        if args.read_from_local == 'False':
            return {'preds': pred_dict, 'url_mapper': {key: model.url_mapper[key] for key in pred_dict}}
        else:
            return {'preds': pred_dict}


if __name__ == '__main__':
    st = time.time()
    return_dict = main()
    print(return_dict)
    end = time.time()
    print('Time:', round(end - st))
