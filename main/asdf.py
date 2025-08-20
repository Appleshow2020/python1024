import sys
import copy
from collections import deque

input = sys.stdin.readline

r, c = map(int, input().split())
l = [list(input().rstrip()) for _ in range(r)]
n = int(input())
h = list(map(int, input().split()))
turn = False


def bfs(l):
    for row in range(r):
        for col in range(c):
            if l[row][col] == 'x':
                return (row, col)
    return (-1, -1)


def dfs(cur):
    global ll, newqueue
    dxy = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for i in range(4):
        nx = cur[0] + dxy[i][0]
        ny = cur[1] + dxy[i][1]

        if 0 <= nx < r and 0 <= ny < c:
            if not visited[nx][ny] and ll[nx][ny] == 'x':
                visited[nx][ny] = True
                ll[nx][ny] = '.'
                newqueue.append((nx,ny))
                dfs((nx, ny))

def dfs2(graph, cur, r,c):
    dxy = [(0,1),(0,-1),(1,0),(-1,0)]
    for i in range(4):
        nx=cur[0]+dxy[i][0]
        ny=cur[1]+dxy[i][1]
        if 0<=nx<r and 0<=ny<c:
            if not visited[nx][ny] and graph[nx][ny] == 'x':
                visited[nx][ny] = True
                if nx == r-1:
                    return True
                dfs2(graph,(nx,ny),r,c)

def drop(l, r1, r2):
    global queue
    visited = [[False] * c for _ in range(r)]
    queue = deque()

    def dropdfs(graph, cur, r, c, Flag, root):
        global queue
        dxy = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for i in range(4):
            nx = cur[0] + dxy[i][0]
            ny = cur[1] + dxy[i][1]
            
            if 0 <= nx < r and 0 <= ny < c and not Flag:
                if not visited[nx][ny] and graph[nx][ny] == 'x':
                    visited[nx][ny] = True
                    if nx != r-1:
                        queue.append((nx,ny))
                    else:
                        queue = deque()
                        queue.append(root)
                        Flag = True
                    dropdfs(graph,(nx,ny),r,c,Flag,root)
    
    def dropcluster(cluster, grid):
        for x, y in cluster:
            grid[x][y] = '.'

        fall_dist = r
        for x, y in cluster:
            nx = x + 1
            dist = 0
            while nx<r and grid[nx][y] == '.':
                nx += 1
                dist += 1
            fall_dist = min(fall_dist, dist)

        for x, y in sorted(cluster,reverse=True):
            grid[x+fall_dist][y] = 'x'
    
    queue.append(r1)
    dropdfs(l,r1,r,c,False,r1)
    if len(queue)>1:
        dropcluster(queue, l)
    else:
        queue = deque()
        queue.append(r2)
        dropdfs(l,r2,r,c,False,r2)
        if len(queue)>1:
            dropcluster(queue,l)    


    return l


for idx in range(n):
    height = h[idx]
    row = r - height
    found = False
    if not turn:
        for col in range(c):
            if l[row][col] == 'x':
                l[row][col] = '.'
                found = True
                break
    else:
        for col in range(c - 1, -1, -1):
            if l[row][col] == 'x':
                l[row][col] = '.'
                found = True
                break
    turn = not turn

    if not found:
        continue

    newqueue = deque()
    ll = copy.deepcopy(l)
    visited = [[False] * c for _ in range(r)]
    root = bfs(l)
    dfs(root)
    root2 = bfs(ll)
    # if root2 != (-1, -1) and root != root2:
    #     l = drop(l, root, root2)
    l = drop(l, root,root2)
        



for i in range(r):
    print("".join(l[i]))