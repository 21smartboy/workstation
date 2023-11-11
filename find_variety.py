def find_variety(label_set,target)
    selected_item[]
    for item in label_set:
        if item.startwith(target):
            selected_item.append(item)
    return selected_item
