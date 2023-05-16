RowSize = 8
ColSize = 9
for col in range(ColSize):
  for i in range(RowSize):
    id = RowSize * col + i
    if col < ColSize - 1:
      if i < RowSize - 1:
        print(f'router {id} node {id} router {id+1} router {id+RowSize}')
      else:
        print(f'router {id} node {id} router {id+RowSize}')
    else:
      if i < RowSize - 1:
        print(f'router {id} node {id} router {id+1}')
      else:
        print(f'router {id} node {id}')