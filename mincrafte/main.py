# 기존 코드
from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
import json
import gc

gc.collect()

# ------------------------------ MODULES ------------------------------

"""

공통 TODO : 렉 해소, 딜레이 구현

전반적으로 렉이 심함.

에셋....? 은 실제 마크 에셋으로 대체.

"""

# --------------------------------------------------------
# CONSTANTS

app = Ursina()

player = FirstPersonController()

window.fps_counter.enabled = False
window.exit_button.visible = False
window.borderless = False

JUMP_HEIGHT = 2 # Default: 2
JUMP_DURATION = 0.5 # Default: 0.5
jump_fall_after = 0.35 # Default: 0.35
GRAVITY_SCALE = 1 # Default: 1
MOUSE_SENSITIVITY = Vec2(40,40) # Default: (40,40)
RUN_SPEED = 5 # Default: 5
MAP_SIZE = 2**31 - 1

player.jump_height = JUMP_HEIGHT
player.jump_up_duration = JUMP_DURATION
player.mouse_sensitivity = MOUSE_SENSITIVITY
player.speed = RUN_SPEED
player.gravity = GRAVITY_SCALE

# --------------------------------------------------------
# VARIABLES

punch = Audio('assets/punch', autoplay=False)

blocks = {
    "none":load_texture("assets/none.png"),
    "grass_block":load_texture("assets/blocks/grass_block.png"),
    "dirt":load_texture("assets/blocks/dirt_block.png")
}

hotbar = [None,
    blocks["grass_block"],
    blocks["dirt"]
]
hotbar += [blocks['none']]*(10-len(hotbar))
hotbar_id = 1

with open(".\\assets\\json\\delay.json",'r') as f:
    delay = json.load(f)

sky = Entity(
    parent=scene,
    model='sphere',
    texture=load_texture('assets/sky.jpg'),
    scale=500,
    double_sided=True
)

hand = Entity(
    parent=camera.ui,
    model='assets/block',
    texture=hotbar[hotbar_id],
    scale=0.2,
    rotation=Vec3(-10, -10, 10),
    position=Vec2(0.6, -0.6)
)

def update():
    global chunk
    _chunk = Chunk(player)
    nchunk = _chunk.chunk

    if nchunk != chunk:
        chunk = nchunk
        gbc.generate_chunk(chunk)
    player_x, _, player_z = player.get_position()
    for dx in range(-1, 2):
        for dz in range(-1, 2):
            neighbor_chunk = (chunk[0] + dx, chunk[1] + dz)
            neighbor_chunk_center_x = (neighbor_chunk[0] * 16)
            neighbor_chunk_center_z = (neighbor_chunk[1] * 16)

            if neighbor_chunk not in gbc.generated_chunks and abs(player_x - neighbor_chunk_center_x)-8 <= 4 and abs(player_z - neighbor_chunk_center_z)-8 <= 4:
                gbc.generate_chunk(neighbor_chunk)

    if held_keys['left mouse'] or held_keys['right mouse']:
        punch.play()
        hand.position = Vec2(0.4, -0.5)
    else:
        hand.position = Vec2(0.6, -0.6)

class Chunk:
    def __init__(self,player):
        self.x, self.z =player.x,player.z

    @property
    def chunk_x(self):
        return (self.x) // 16 - (1 if self.x < 0 and (self.x % 16) != 0 else 0)

    @property
    def chunk_z(self):
        return (self.z) // 16 - (1 if self.z < 0 and (self.z % 16) != 0 else 0)

    @property
    def chunk(self):
        return int(self.chunk_x), int(self.chunk_z)

_chunk = Chunk(player)
chunk = _chunk.chunk

class GenerateByChunk:
    def __init__(self):
        self.generated_chunks = set()
        self.storaged_chunks = set()
        directory = "./assets/map"
        # 지정된 디렉토리에서 모든 청크 파일 검색
        if os.path.exists(directory):
            for file_name in os.listdir(directory):
                if file_name.startswith("c.") and file_name.endswith(".json"):
                    try:
                        # 파일 이름에서 청크 좌표 추출
                        _, cx, cy, _ = file_name.split(".")
                        cx, cy = int(cx), int(cy)
                        self.storaged_chunks.add((cx, cy))
                    except ValueError:
                        # 파일 이름이 예상 형식이 아닌 경우 무시
                        continue

    def generate_chunk(self, chunk):
        if chunk in self.generated_chunks:
            return
        if chunk not in self.storaged_chunks:
            from collections import defaultdict
            cx,cy = chunk[0],chunk[1]
            # 데이터 저장용 딕셔너리
            data = defaultdict(lambda: defaultdict(dict))
            # 데이터 작성
            for x in range(-8, 9):
                for z in range(-8, 9):
                    key = f"{16*cx+x},{16*cy+z}"
                    data[key][0] = {
                        "block": "grass_block"
                    }
            for x in range(-8, 9):
                for y in range(-1, -3, -1):
                    for z in range(-8, 9):
                        key = f"{16*cx+x},{16*cy+z}"
                        data[key][y] = {
                            "block": "dirt"
                        }
            # 파일 경로 설정
            dir_path = f".\\assets\\map\\c.{cx}.{cy}.json"
            os.makedirs(os.path.dirname(dir_path), exist_ok=True)
            # JSON 파일 쓰기
            with open(dir_path, 'w') as f:
                json.dump(data, f, indent=4)
        print(chunk)
        with open(f".\\assets\\map\\c.{chunk[0]}.{chunk[1]}.json", "r") as f:
            region = json.load(f)
        
        for xz, value in region.items():
            for y, value1 in value.items():
                x, z = map(int, xz.split(","))
                Block((x, int(y), z), f'assets/blocks/{value1["block"]}.png')
                
        self.generated_chunks.add(chunk)

        return

gbc = GenerateByChunk()

class Block(Button):
    """
    일반 블럭 클래스
    args : 위치(tuple[3]), 텍스쳐(str(assets/blocks/?.png))
    """
    def __init__(self, position=(0, 0, 0), texture='assets/none.png'):
        super().__init__(
            parent=scene,
            position=position,
            model='assets/block',
            origin_y=0.5,
            texture=texture,
            color=color.color(0, 0, random.uniform(0.9, 1.0)),
            scale=0.5
        )

    def input(self, key):
        if self.hovered:
            if key == 'right mouse down':
                Block(position=self.position + mouse.normal, texture=hotbar[hotbar_id])
            elif key == 'left mouse down':
                destroy(self)

class Inventory(Entity):
    """
    인벤토리
    TODO:UI 개선, 인벤토리 활성화 시 화면 고정
    """

    def __init__(self):
        super().__init__(parent=camera.ui)
        self.items = {}  # 인벤토리에 들어갈 아이템 리스트

        # 전체 배경
        self.background = Entity(
            parent=self,
            model='quad',
            color=color.rgb(198,198,198,255),
            scale=(1.2, 0.9),
            position=(0, 0)
        )
        self.k = 0.15
        for temp, position in enumerate([(0, 0.225 + self.k), (0, 0.1 + self.k), (0, 0.1 - 0.125 + self.k), (0, 0.1 - (0.125 * 2) + self.k)]):
            slot = Button(
                parent=self,
                model='quad',
                color=color.gray,
                scale=(0.1, 0.1),
                position=(-0.5 + position[0], position[1]),
            )
            self.items[103 - temp] = slot

        # 제작 슬롯 (오른쪽 상단)
        self.crafting_slots = []
        for y in range(2):
            for x in range(2):
                slot = Button(
                    parent=self,
                    model='quad',
                    color=color.gray,
                    scale=(0.1, 0.1),
                    position=(0.1 + x * 0.12, 0.3 - y * 0.15),
                )
                self.crafting_slots.append(slot)

        # 결과 슬롯
        self.result_slot = Button(
            parent=self,
            model='quad',
            color=color.gray,
            scale=(0.1, 0.1),
            position=(0.45, 0.2),
        )
        d = 0.025
        for y in range(3):
            for x in range(9):
                slot = Button(
                    parent=self,
                    model='quad',
                    color=color.gray,
                    scale=(0.1, 0.1),
                    position=(-0.5 + x * 0.1 + d * x, -0.15 - y * 0.12),
                    on_click=self.select_slot
                )
                slot.item_count = Text(
                    parent=slot,
                    text='',
                    scale=1.5,
                    position=(0.03, -0.03),
                    color=color.black
                )
                self.items[9 * (y + 1) + x] = slot

    def add_item(self, item_texture, count=1):
        for slot in self.slots:
            if not slot.texture:  # 빈 슬롯을 찾음
                slot.texture = item_texture
                slot.item_count.text = str(count)
                self.items.append((item_texture, count))
                return

    def select_slot(self):
        self.color = color.azure

inventory = Inventory()
inventory.enabled = False

def input(key):
    global hotbar_id, hand
    global chunk

    if key.isdigit():
        hotbar_id = int(key)
        if hotbar_id > 9:
            hotbar_id = 1
        elif hotbar_id < 1:
            hotbar_id = 9

        hand.texture = hotbar[hotbar_id]

    if key == 'e':
        inventory.enabled = not inventory.enabled
        mouse.locked = not inventory.enabled
        player.speed = 0 if inventory.enabled else RUN_SPEED

chunk = _chunk.chunk
gbc.generate_chunk(chunk)
player.set_position((0, 0, 0))
mouse.locked = True

app.run()
