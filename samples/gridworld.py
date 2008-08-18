# This code is so you can run the samples without installing the package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
#

import reply
import numpy
from numpy import array
import random
debug = 0

class GridWorld(reply.World):
    ENTER, EXIT, MINE = range(3)
    
    def __init__(self, size, enter, exits, mines):
        """
        size: tuple for the world size
        enter: tuple coordinate for the point of enter
        exit: tuple coordinate for the exit points
        mines: list of tuple coordinates for the mines positions
        """
        self.size = size
        self.enter = enter
        self.exits = exits
        self.mines = mines
        
        self.position = self.enter
        

    def get_action_space(self):
        return [ reply.dimension(0,3,4)  ]
    
    def get_state_space(self):
        xz, yz = self.size
        return [ reply.dimension(0,xz-1,xz), reply.dimension(0,yz-1,yz), ]
        
    def get_reward(self, state):
        if self.position in self.exits:
            return 10
        if self.position in self.mines:
            return -1
        return -0.01
    
        
    def is_final(self, state):
        if self.position in self.exits:
            return True
        return False
    
    def move(self, position, dir):
        if dir == 0: # up
            p = position[0], (position[1]+1)%self.size[1]
        elif dir == 1: # down
            p = position[0], (position[1]-1)%self.size[1]
        elif dir == 2: # left
            p = (position[0]-1)%self.size[0], position[1]
        elif dir == 3: # right
            p = (position[0]+1)%self.size[0], position[1]
        return p
    
    def do_action(self, solver, action):
        dir = action[0]
        if debug:
            print "at", self.position,
            print "with action", dir,
        self.position = self.move( self.position, dir )
        self.trace.append( self.position )
        if debug:
            print "moved to", self.position,
            if self.position in self.exits:
                print "and exited"
            if self.position in self.mines:
                print "and found a MINE"
            else:
                print
        
    
            
        
        
    def get_initial_state(self):
        self.trace = [self.enter]
        self.position = self.enter
        return self.get_state()
    
    def get_state(self):
        return self.position
        
    
if __name__ == "__main__":
    import pyglet
    from pyglet import gl
    
    g = GridWorld((10,10), (1,1), [(5,5)], 
                  [
                   (0,0), (1,0), (2,0), 
                   (0,1),
                   (0,2), (1,2), (2,2), 
                   (3,4), (4,4), (5,4), (6,4),
                   (3,5),
                   (3,6), (4,6), (5,6), (6,6)
               ])
    space = g.get_state_space(), g.get_action_space()
    encoder = reply.encoder.DistanceEncoder(*space)
    storage = reply.storage.TableStorage(encoder)
    #learner = reply.learner.QLearner(1, 0.01,0.99, 0.05)
    learner = reply.learner.SarsaLearner(1, 0.01,0.99, 0.05)
    selector = reply.selector.EGreedySelector(0.1, 1)
    
    r = reply.RL(
            learner, 
            storage,
            encoder, 
            selector
        )
    delta = 0
    cellsize = 24 + delta * 2
    sx, sy = g.size
    window = pyglet.window.Window(width=max(200, sx*cellsize), height=sy*cellsize+100)
    up = pyglet.resource.image('up.png')
    down = pyglet.resource.image('down.png')
    left = pyglet.resource.image('left.png')
    right = pyglet.resource.image('right.png')
    exit = pyglet.resource.image('exit.png')
    mine = pyglet.resource.image('mine.png')
    

    class Mouse: pass
    mouse = Mouse()
    mouse.x = mouse.y = 0
    @window.event
    def on_mouse_motion(x, y, dx, dy):
        mouse.x = x
        mouse.y = y
        
    def get_path():
        current = g.enter
        visit = {}
        path = []
        while not current in visit:
            path.append( current )
            visit[current]=1
            current = g.move( current, numpy.argmax(storage.get_state_values( encoder.encode_state( current ) )) )
            if current in g.exits:
                path.append(current)
                break
        return path
            
    @window.event
    def on_draw():
        window.clear()
        #paint white BG
        gl.glColor4f(1,1,1,1) 
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f( 0, 0 )
        gl.glVertex2f( 0, window.height )
        gl.glVertex2f( window.width, window.height )
        gl.glVertex2f( window.width, 0 )
        gl.glEnd()
        

        # draw cell values
        x = mouse.x // cellsize
        y = mouse.y // cellsize
        if x < sx and y < sy:
            vals = storage.get_state_values( encoder.encode_state( (x,y) ) )
            label = pyglet.text.HTMLLabel(
                '<pre>cell %i,%i:\nup   : %e\ndown : %e\nleft : %e\nright: %e</font></pre>'%(x,y, vals[0], vals[1], vals[2], vals[3]),
                          x=0, y=window.height,multiline=True, width=window.width, anchor_y='top'
                          )
            label.draw()

        # draw cell direction
        for x in range(sx):
            for y in range(sy):
                if (x,y) in g.mines:
                    img = mine
                elif (x,y) in g.exits:
                    img = exit
                else:
                    img = [up, down, left, right][
                        numpy.argmax(storage.get_state_values( encoder.encode_state( (x,y) ) ))
                        ]
                img.blit(x*(cellsize)+delta, y*cellsize+delta)
                
        # draw best path
        path = get_path()
        gl.glLineWidth(2)
        gl.glColor4f(0,255,0,255)
        gl.glBegin(gl.GL_LINE_STRIP)
        for x,y in path:
            gl.glVertex2f( x*cellsize+cellsize/2, y*cellsize+cellsize/2 )
        gl.glEnd()
        gl.glColor4f(1,1,1,1) 

            
            
    episode = 0
    def step(dt):
        global episode
        total_reward,steps  = r.run(g)
        err = 0
        print 'Espisode: ',r.episodes,'  Steps:',steps,'  Reward:',total_reward, "alpha", learner.alpha
        episode += 1

    
    
    pyglet.clock.schedule_interval(step, 0.01)
    pyglet.app.run()