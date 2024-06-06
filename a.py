status = False
where = 0

sl = int(input())
s = str(input())


for i in s:
    if where>=2:
        print(5) if status == False else print(1)
        break
    if i == "P":
        if status:
            status = False
        else:
            status = True
    else:
        where+=1
if where != 2:
    print(0)