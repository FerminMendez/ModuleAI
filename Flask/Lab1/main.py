import json
from flask import Flask
from flask_restful import Resource, Api, reqparse, abort
# main.py
from controllers import Index, AllVideos, IdVideo, VideoSchedule

from controllers import write_changes_to_file

# Configuración del servidor
app = Flask("VideoAPI")
api = Api(app)

# Create inital file
# write_changes_to_file()
# Routes
with open('videos.json', 'r') as f:
    videos = json.load(f)

api.add_resource(Index, "/")
api.add_resource(AllVideos, "/videos")
api.add_resource(IdVideo, "/videos/<string:video_id>")
api.add_resource(VideoSchedule, "/videos")

# api.add_resource(CONTROLLER CLASS FROM CONTROLLERS, "/rout/<string:param>")

# Configuración del servidor
if __name__ == "__main__":
    app.run()
