# Steps to run (WSL-Linux)

## Initial Setup
* Change directory to where project files exist
  * `cd ./image_classifier`

* Install virtualenv
  * `sudo pip3 install virtualenv`

* Generate environment
  * `virtualenv classifier_env --python=python3.8`

* Install requirements
  * `pip install -r ./requirements.txt`

## Run
* Activate the environment
  * `source classifier_env/bin/activate`

* Run it by giving list of URLs to download
  * `pyton main.py -url <URLS>`
  * Example: `python main.py -url https://picsum.photos/200 https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500`

* Run it by giving the path of JSON file that contains URLs
  * `python main.py -jp <JSON PATH>`
  * `Example: python main.py -jp test_url.json`

* If you already have the images saved in the machine
  * `python main.py -id <Directory of the images> -rl True`
  
## Models

Models are too big to add to the git, please download the models following this [Link](https://drive.google.com/file/d/1ZOaeV6jwUoDfKI3Zd2CO_jVXix1i8-gQ/view?usp=share_link)

Unzip the above link to the repo before running the code

# Results
 * If its given list of URLs or path to json file that contains URLs, it returns to 2 dictionaries:
     * prediction dictionary which maps file names to predictions
     * url dictionary which maps filenames to URLs
