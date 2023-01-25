# Simple ML multiclass- classification service

## About the project
This project consists of the Machine Learning service for image
classification task, created with FastAPI. Image classification can be
performed through **POST** method to created API as well as using local python
script. The project runs on Docker containers and is managed by Docker-Compose.
Additionally, api service is connected with Celery worker and Redis broker,
to perform the inference process in queued tasks.

**ATTENTION**
Dataset images have to be downloaded from the [Source](http://vision.stanford.edu/aditya86/ImageNetDogs/) and unpacked in the `dataset/`
so the project won't crash at the start.

## Dataset & model
The task is defined as the multi-label-classification problem. The chosen
CNN model consists of the base model, which is pre-trained **EfficientNet_V2** network
and head model - Sequential model composed of FeedForward Neural Networks.

Dataset for this task consists of 20,580 images of different dog breeds, so
the task is 120-class classification problem.

Currently, saved "best" model performs with ~65% accuracy on validation set, but training
with larger epochs can probably increase this result.

## Functionality

### Starting app on docker
To start the docker-compose application we can simply run (being located in project dir)
in terminal:
```shell
docker-compose up --rebuild
```
or equivalently with make command (using Makefile):
```shell
make build
```
After build is completed we should see then logs of started three services: web, celery and redis.
The API should be accessible on localhost with port 8000 (check logs), e.g. http://0.0.0.0:8000.

### Making predictions
The simplest way to start making predictions with API is to use automatically created docs:
1. Go on http://0.0.0.0:8000/docs
2. Check `api/predict` endpoint and click **Try it out** button.
3. Select the image to upload
4. You can also pass optional device argument ["cpu", "gpu", "mps"] and the inference will be performed on it if the device is accessible, otherwise it will run on default device.
5. Click **Execute** and check the response.

### Celery task status
After making predictions, we can check in celery logs (`docker-compose logs -f celery`)
that the action was registered as task and sent to celery queue. We can also check
the task status passing task_id to http://0.0.0.0:8000/api/tasks/{task_id}.


### Local scripts
Locally we can run inference as well as network training by running python scripts
located in `src/scripts/`: `inference.py` and `model_training.py` using make command (with default settings):

```shell
make predict
```
```shell
make train-model
```
or using docker-compose. Example below runs training with applied head model on 10 epochs with batch size = 64 for ResNet model. The other available model is `"EfficientNet"` and with save model option.
```shell
docker-compose exec web python src/model_training.py --epochs=10 --batch=64 --model="ResNet" --save --extractor
```
So model training can be run with the following settings:
- epochs, batch, learning rate
- different base model architecture: ResNet or EfficientNet
- different type of architecture functionality: with head model (so the base weights are frozen) or in the fine-tuning mode.

### Other
We can check model training data as Loss value, Accuracy and F_score both on training and test dataset
by checking: http://0.0.0.0:8000/api/stats/

Other functionalities defined in Makefile include e.g. code reformatting using black and isort.

## TO-DO
- [ ] Increase model performance
- [ ] Add tests
- [ ] Make prediction visualizations available from API
