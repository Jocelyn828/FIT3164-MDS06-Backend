from django.db import connection

def reset_sequence(model):
    # Used to reset the primary key sequence of a model (id field) to 1
    # when deleting all objects from the table.
    table_name = model._meta.db_table
    if connection.vendor == 'postgresql':
        sql = f"ALTER SEQUENCE {table_name}_id_seq RESTART WITH 1"
    elif connection.vendor == 'sqlite':
        sql = f"DELETE FROM sqlite_sequence WHERE name='{table_name}'"
    elif connection.vendor == 'mysql':
        sql = f"ALTER TABLE {table_name} AUTO_INCREMENT = 1"
    else:
        return  # Unsupported DB backend
    
    with connection.cursor() as cursor:
        cursor.execute(sql)
