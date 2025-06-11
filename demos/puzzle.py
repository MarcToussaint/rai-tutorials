import robotic as ry
import numpy as np
import time

def dist(C: ry.Config, a: ry.Frame, b: ry.Frame):
    y, _ = C.eval(ry.FS.negDistance, [a.name, b.name])
    return -y[0]

def above(C: ry.Config, a: ry.Frame, b: ry.Frame):
    y, _ = C.eval(ry.FS.aboveBox, [a.name, b.name])
    return np.max(y)

ry.params_add({'botsim/verbose': 0})

def test():
    C = ry.Config()
    C.addFile('puzzle.g')
    C.view(False)

    bot = ry.BotOp(C, False)
    bot.home(C)

    C.get_viewer().setupEventHandler(True)

    cursor = C.addFrame('cursor')
    ego = C.getFrame("ego")
    nogo = C.getFrame("nogo")
    objs = []
    for f in C.getFrames():
        if f!=ego and 'logical' in f.getAttributes():
            objs.append(f)
            print(f.name)

    bot.wait(C)
    cursorLocked=False
    grasped: ry.Frame = None
    glued = []
    stop = False
    
    while(not stop):
        bot.sync(C, -1.)
        time.sleep(.02)

        q = bot.get_q()
        ctrlTime = bot.get_t()

        pos = C.get_viewer().getEventCursor()

        if pos.size==6:
            cursor.setPosition(pos[:3])
            delta = pos[:2] - q

            l = np.linalg.norm(delta)
            if not cursorLocked and l<.2:
                cursorLocked=True

            isNoGo = above(C, cursor, nogo)<0

            if cursorLocked and not isNoGo:
                cap = .5
                if l>cap:
                    delta *= cap/l
                q += delta
                bot.move(q.reshape((1,-1)), [.1], True, ctrlTime)

        events = C.get_viewer().getEvents()

        for e in events:
            if e=="key down space":
                dists = [dist(C, ego, f) for f in objs]
                nearest = objs[np.argmin(dists)]

                print(f'nearest: {nearest.name}')
                bot.attach("ego", nearest.name)
                grasped=nearest
                grasped.setColor([1.,.5,0])

            if e=="key up space" and grasped is not None:
                bot.detach(grasped.name)
                grasped.setColor([.9])
                grasped=0

            if e=="key down tab": #glue
                dists = [dist(C, ego, f) for f in objs]
                idx = np.argpartition(dists, 2)
                nearest = [objs[idx[0]], objs[idx[1]]]
                d = dist(C, nearest[0], nearest[1])
                print(f'nearest objects: {nearest[0].name} {nearest[1].name} dist: {d}')
                if d<.1:
                    if (nearest[0],nearest[1]) in glued:
                        print('un-gluing')
                        bot.detach(nearest[1].name)
                        nearest[0].setColor([.9])
                        nearest[1].setColor([.9])
                        glued.remove((nearest[0],nearest[1]))
                    elif (nearest[1],nearest[0]) in glued:
                        print('un-gluing')
                        bot.detach(nearest[0].name)
                        nearest[0].setColor([.9])
                        nearest[1].setColor([.9])
                        glued.remove((nearest[1],nearest[0]))
                    else:
                        print('gluing')
                        glued.append((nearest[0], nearest[1]))
                        bot.attach(nearest[0].name, nearest[1].name)
                        nearest[0].setColor([0.,.5,1.])
                        nearest[1].setColor([0.,.5,1.])
                else:
                    print('no 2 nearest objects found')

            if e=="key down q": #quit
                stop=True
                break

if __name__ == '__main__':
    test()
