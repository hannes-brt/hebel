from mongoengine import Document, EmbeddedDocument, StringField, \
     ObjectIdField, IntField, ListField, FloatField, ReferenceField, \
     DateTimeField, EmbeddedDocumentField, connect
from mongoengine.base import BaseField
from datetime import timedelta
from . import configmodule as config

connect(config.DB_NAME, host=config.MONGO_DB_URL)

class TimedeltaField(BaseField):
    """A timedelta field.                                                                                        
                                                                                                                 
    Looks to the outside world like a datatime.timedelta, but stores                                             
    in the database as an integer (or float) number of seconds.                                                  
                                                                                                                 
    """
    def validate(self, value):
        if not isinstance(value, (timedelta, int, float)):
            self.error(u'cannot parse timedelta "%r"' % value)

    def to_mongo(self, value):
        return self.prepare_query_value(None, value)

    def to_python(self, value):
        return timedelta(seconds=value)

    def prepare_query_value(self, op, value):
        if value is None:
            return value
        if isinstance(value, timedelta):
            return value.total_seconds()
        if isinstance(value, (int, float)):
            return value

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
    runtime = TimedeltaField()
    date_finished = DateTimeField()
    
