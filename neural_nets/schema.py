from mongoengine import Document, EmbeddedDocument, StringField, \
     ObjectIdField, IntField, ListField, FloatField, ReferenceField, \
     DateTimeField, EmbeddedDocumentField, connect
from . import configmodule as config

connect(config.DB_NAME, host=config.MONGO_DB_URL)

class Dataset(Document):
    name = StringField(primary_key=True)

class ErrorLog(EmbeddedDocument):
    epoch = ListField(IntField(), default=list)
    value = ListField(FloatField(), default=list)

class Experiment(Document):
    task_id = StringField(primary_key=True)
    name = StringField()
    yaml_config = StringField()
    model_checksum = StringField()
    train_error = EmbeddedDocumentField(ErrorLog)
    validation_error = EmbeddedDocumentField(ErrorLog)
    epochs = IntField()
    test_error = FloatField()
    dataset = ReferenceField(Dataset)
    save_path = StringField()
    runtime = DateTimeField()
    date_finished = DateTimeField()

