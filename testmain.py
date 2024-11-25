from fastapi import FastAPI
from testapi import router
from testapidb import base, engine

base.metadata.create_all(bind = engine)
app = FastAPI()
app.include_router(router)