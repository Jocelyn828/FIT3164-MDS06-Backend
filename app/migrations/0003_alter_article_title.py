# Generated by Django 4.2.15 on 2025-03-18 09:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_article'),
    ]

    operations = [
        migrations.AlterField(
            model_name='article',
            name='title',
            field=models.TextField(blank=True),
        ),
    ]
