"""This toll will establish an ssh tunnel and allow us to push SQL queries onto the closter from within our python environment."""

import psycopg2
from sshtunnel import SSHTunnelForwarder
import logging

logger = logging.getLogger('airevo') 

try:
    with SSHTunnelForwarder(
         ('app.getroundto.it', 22),
         #ssh_private_key="</path/to/prcond aivate/ssh/key>",
         ssh_username="joseph",
         ssh_password="catch-PRODUCE-bill-FARM", 
         remote_bind_address=("192.168.1.60", 5432)) as server:
         
         server.start()
         logger.debug("server connected")

         params = {
             'database': 'su_dev',
             'user': 'joseph',
             'password': 'public-INSECTS-rule-TASTE',
             'host': 'localhost',
             'port': server.local_bind_port
             }

         conn = psycopg2.connect(**params) #establishes connection
         curs = conn.cursor() #creates a cursor to interact with our database
         logger.debug("database connected")

         command = "SELECT * FROM joseph.pnrs LIMIT 10;" 
         curs.execute(command) #executes our command
         res = curs.fetchall() #fetch the data loaded in cursor can also use curs.fetchall()
         print(res)
         logger.info("Done")

except Exception as ex:
    logger.fatal("Connection Failed")
    raise ex