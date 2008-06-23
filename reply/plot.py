import Queue
import pylab
import numpy
import threading
import time

class LinePlot:
    def __init__(self, figure=1, subplot=111):
        pylab.ion()
        self.data = numpy.array([])
        self.axis = numpy.array([])
        self.count = 0
        self.min = 0
        self.max = 0
        self.min_framerate = 0.5
        
        self.figure = pylab.figure(figure)
        self.subplot = pylab.subplot(subplot)
        self.lines = pylab.plot([0])

        self.pending = Queue.Queue()
        self.t = threading.Thread(target=self.update_plot)
        self.t.setDaemon(True)
        self.t.start()
    
        self.t2 = threading.Thread(target=self.draw)
        self.t2.setDaemon(True)
        #self.t2.start()
    
    def draw(self):
        while True:
            pylab.draw()
            time.sleep(0.25)
    def add(self, value):
        self.pending.put(value)
        
    def update_plot(self):
        
        pending = []
        while True:
            
            if not pending:
                next = self.pending.get()
                first_time = time.time()
                pending.append( next )
            else:
                waited = time.time()-first_time
                if waited < self.min_framerate and not len(pending)>100:
                    try:
                        next = self.pending.get(timeout=self.min_framerate-waited)
                        pending.append( next )
                    except Queue.Empty:
                        pass
                else:
                    print "drawing", len(pending)
                    
                    self.min = m = min( self.min, min( pending ) )
                    self.max = M = max( self.max, max( pending ) )
                    
                    self.data = numpy.hstack([self.data, numpy.array(pending)])
                    self.axis = numpy.hstack([self.axis, numpy.array(range(self.count, self.count+len(pending)))])
                    self.lines[0].set_data(self.axis, self.data)
                    self.subplot.set_ylim((m,M))
                    self.count += len(pending)
                    self.subplot.set_xlim((0,self.count))
                    pylab.draw()
                    pending = []

        

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
        
if __name__=="__main__":
    import random
    l = LinePlot(1)
    l2 = LinePlot(2)
    for i in range(1000):
        l.add( i + random.random()*i)
        l2.add( i + random.random()*i*2)
    raw_input("done?")
    print "done."