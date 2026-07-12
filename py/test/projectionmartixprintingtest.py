from utils.printing import *

camera_params = {0:{"P":[],"id":0},1:{"P":[],"id":1}}
def print_projection_matrices():
    printf("",ptype=LT.info,end=" ",useReset=False)
    for cam_id, data in camera_params.items():
        print(f"{cam_id} Projection Matrix:\n{data['P']}")
    print(Colors.reset)

print_projection_matrices()