from django.db import models


class Document(models.Model):
    docfile = models.FileField(upload_to='documents/%Y/%m/%d')


class OutputDocument(models.Model):
    documents = models.ManyToManyField(Document, blank=False)
    output = models.FileField(upload_to='output/%Y/%m/%d')
