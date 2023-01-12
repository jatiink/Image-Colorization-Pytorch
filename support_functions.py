def board_add_images(board, tag_name, img_list, step_count, names=None):
    if names is None:
        for i, img in enumerate(img_list):
            board.add_image(f'{tag_name}/{i}', img, step_count, dataformats='HWC')
            board.flush()
    else:
        for name, img in zip(names, img_list):
            board.add_image(f'{tag_name}/{name}', img, step_count, dataformats='HWC')
            board.flush()