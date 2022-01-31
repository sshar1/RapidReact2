# Writing the Tornado server to initiate the client request

import tornado.ioloop
import tornado.web
import tornado.websocket
import cv2 as cv

from processing.CargoDetection import detectCargo, source
from processing.TapeDetection import detect_line
from processing.img_to_str import to_b64

from tornado.options import define, options

# Websocket defined with port 2601
define('port', default = 1234, type=int)

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

    # function to respond to a message on the WebSocket
    def on_message(self, message):
        _, frame = cap.read()
        
        # enter open cv code here
        output_image = detectCargo(frame, message)
        cv.imwrite("./websocket/frame.jpg", output_image)

        self.write_message(to_b64("./websocket/frame.jpg"))

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
        cv.imwrite("./websocket/line.jpg", output_image)
        self.write_message(to_b64("./websocket/line.jpg"))


    def on_close(self):
        print('connection closed')

app = tornado.web.Application([
    (r'/', IndexHandler),
    (r'/ws/', WebSocketHandler),
    (r'/line/', ShadowHandler),
    (r'/line/ws/', ShadowSocketHandler)
])

if __name__ == '__main__':
    app.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()