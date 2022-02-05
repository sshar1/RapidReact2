# Writing the Tornado server to initiate the client request
import tornado.ioloop
import tornado.web
import tornado.websocket
import cv2 as cv

from processing.CargoDetection import detectCargo, source, sendData
from processing.TapeDetection import detect_line
from processing.img_to_str import to_b64

from tornado.options import define, options

# Websocket defined with port 2601
define('port', default = 2601, type = int)
cap = cv.VideoCapture(source, cv.CAP_DSHOW)

# LIGHTING: -1 for internal camera, -7 for FISHEYE, -4 for Microsoft HD-3000
cap.set(cv.CAP_PROP_EXPOSURE, -7)

# This handler handles a call to the base of the server \
# (127.0.0.1:8888/ -> 127.0.0.1:8888/index.html)
class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('./websocket/www/index.html')

# This handler handles a websocket connection
class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self, *args):
        print('new cargo connection!')

    def on_message(self, message):
        _, frame = cap.read()
        
        output_image = detectCargo(frame, message)
        cv.imwrite("./websocket/cargoFrame.jpg", output_image)

        self.write_message(to_b64("./websocket/cargoFrame.jpg"))

    def on_close(self):
        print('connection closed')


class DataHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("./websocket/www/data.html")

class DataSocketHander(tornado.websocket.WebSocketHandler):
    def open(self, *args):
        print('Data websocket connection')
    
    def on_message(self, message):
        _, frame = cap.read()

        output_image = detectCargo(frame, message)
        cv.imwrite("./websocket/dataFrame.jpg", output_image)

        #self.write_message(to_b64("./websocket/dataFrame.jpg"), sendData(frame, message))
        #self.write_message(to_b64("./websocket/dataFrame.jpg"))
        self.write_message(sendData(frame, message))

    def on_close(self):
        print('connection closed')


class ShadowHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("websocket/www/line.html")

class ShadowSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self, *args):
        print("line websocket connection")
    
    def on_message(self, message):
        _, frame = cap.read()

        output_image = detect_line(frame)
        cv.imwrite("./websocket/lineFrame.jpg", output_image)

        self.write_message(to_b64("./websocket/lineFrame.jpg"))

    def on_close(self):
        print('connection closed')


app = tornado.web.Application([
    (r'/', IndexHandler),
    (r'/ws/', WebSocketHandler),
    (r'/line/', ShadowHandler),
    (r'/line/ws/', ShadowSocketHandler),
    (r'/data/', DataHandler),
    (r'/data/ws/', DataSocketHander)
])

if __name__ == '__main__':
    app.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()