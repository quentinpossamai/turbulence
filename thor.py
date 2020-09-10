# <editor-fold desc="Initiate env">
from ai2thor.controller import Controller
from PIL import Image

RANDOM_SEED = 0

# Initialisation of a controller and in the same time a scene
c = Controller(scene='FloorPlan28', gridSize=0.25, renderDepthImage=False, renderClassImage=False,
               renderObjectImage=False, width=640, height=480, agentMode='bot')

# Initialisation of a scene only
c.reset(scene='FloorPlan28')

# c.step('Initialize', gridSize=0.25, AgentCount=1)

# Place randomly pickupable objects
c.step(action='InitialRandomSpawn', randomSeed=RANDOM_SEED, numPlacementAttemps=5)

# Randomize the state of all the toggleable objects
c.step(action='RandomToggleStateOfAllObjects', randomSeed=RANDOM_SEED)

# Randomize the state of all specific states that can have objects
event = c.step(action='RandomToggleSpecificState', randomSeed=RANDOM_SEED,
               StateChange=['CanOpen', 'CanToggleOnOff', 'CanBeSliced', 'CanBeCooked', 'CanBreak', 'CanBeDirty',
                            'CanBeUsedUp'])
pos = event.metadata['agent']['position']
# </editor-fold>

event = c.step(action='AddThirdPartyCamera', position=dict(x=0, y=0, z=0), fieldOfView=90)

event = c.step(action='Teleport', x=pos['x'], y=pos['y'] + 1, z=pos['z'] - 1)
pos = event.metadata['agent']['position']

event = c.step(action='MoveAhead')

# event = controller.step(action='GetReachablePositions')
# e = event.metadata['actionReturn']

for object_element in event.metadata['objects']:
    print(f'\n\nObject : {object_element}')
    for key, value in object_element.items():
        print(f'Key : {key}, value : {value}')
    break

a = event.frame
b = event.depth_frame
c1 = event.class_segmentation_frame
d = event.instance_segmentation_frame

img1 = Image.fromarray(a, 'RGB')
img1.show()
img2 = Image.fromarray(b, 'RGB')
img2.show()
img3 = Image.fromarray(c1, 'RGB')
img3.show()
img4 = Image.fromarray(d, 'RGB')
img4.show()

c.stop()
