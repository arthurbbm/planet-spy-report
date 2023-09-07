from planet import Auth
from decouple import config

API_USERNAME = config('USER')
API_PASSWORD = config('PASSWORD')

user = API_USERNAME
pw = API_PASSWORD
auth = Auth.from_login(user, pw)
auth.store()
del API_USERNAME, API_PASSWORD, user, pw