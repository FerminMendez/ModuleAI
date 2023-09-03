
from flask import Flask
from flask_restful import Resource, Api, reqparse, abort
#from models import videos
import json


videos = {

    'video1': {'title': 'Hello World in Python', 'uploadDate': 20210917},

    'video2': {'title': 'Why Matlab is the best language Ever', 'uploadDate': 20211117}

}


class Index(Resource):
    def get(self):
        return "Hello World!", 200


class AllVideos(Resource):
    def get(self):

        return videos, 200


class IdVideo(Resource):
    def get(self, video_id):
        allIds = videos.keys()
        if(not video_id in allIds):
            abort(404, message=f"Video {video_id} not found!")
        else:
            return videos[video_id], 200
