import pyglet
from pyglet.gl import *

class Plot:
    def __init__(self, x,y,width, height, 
                xrange, yrange, 
                color = (1,0,0,0),
                title=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.xrange = xrange
        self.xsize = float(xrange[1]-xrange[0])
        self.yrange = yrange
        self.ysize = float(yrange[1]-yrange[0])
        self.color = color
        self.title = title
        
    def start_plot(self):
         pass
         
    def point(self, x, y):
        x_start = (x/self.xsize)*self.width
        x_end = ((x+1)/self.xsize)*self.width
        y_n = (y/self.ysize)*self.height
        
        glColor3f(*self.color)
        glBegin(GL_LINES);
        glVertex2f(x_start, y_n) 
        glVertex2f(x_end, y_n) 
        glEnd( );