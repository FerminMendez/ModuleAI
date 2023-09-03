import json
from flask import Flask
from flask_restful import Resource, Api, reqparse, abort
# main.py
from controllers import Index, AllPokemon
from controllers import write_changes_to_file

# Configuración del servidor
app = Flask("PokemonAPI")
api = Api(app)

# Create inital file
# write_changes_to_file()
# Routes
with open('pokedex.json', 'r') as f:
    pokedex = json.load(f)

api.add_resource(Index, "/")
api.add_resource(AllPokemon, "/all")

# api.add_resource(CONTROLLER CLASS FROM CONTROLLERS, "/rout/<string:param>")

# Configuración del servidor
if __name__ == "__main__":
    app.run()
