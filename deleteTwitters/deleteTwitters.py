import pyautogui
import time

# 设置每次点击之间的延时
delay_between_clicks = 1  # 秒

# 加载按钮的截图（确保这些文件在您的脚本目录中或提供正确的路径）
button_image_1 = R'C:\Users\Desktop\deleteTwitters\1.jpg'  # 第一个按钮的截图
button_image_2 = R'C:\Users\Desktop\deleteTwitters\2.jpg'  # 第二个按钮的截图
button_image_3 = R'C:\Users\Desktop\deleteTwitters\3.jpg'  # 第三个按钮的截图

# 需要删除的项目数量
number_of_items = 1000
time.sleep(5)
for _ in range(number_of_items):
    # 找到并点击第一个按钮
    location = pyautogui.locateCenterOnScreen(button_image_1, confidence=0.8)

    pyautogui.click(location)

    # 等待页面反应
    time.sleep(delay_between_clicks)

    # 找到并点击第二个按钮
    location = pyautogui.locateCenterOnScreen(button_image_2, confidence=0.8)

    pyautogui.click(location)

    # 等待页面反应
    time.sleep(delay_between_clicks)

    # 找到并点击第三个按钮
    location = pyautogui.locateCenterOnScreen(button_image_3, confidence=0.8)

    pyautogui.click(location)

    # 等待页面反应
    time.sleep(delay_between_clicks)









# import pyautogui
# import time
#
# # 设置每次点击之间的延时
# delay_between_clicks = 1  # 秒
#
# # 第一个和第二个按钮的屏幕坐标（两个按钮坐标一致）
# first_and_second_button_coordinates = (1184, 214)
#
# # 第三个确认删除的按钮坐标
# confirm_delete_button_coordinates = (968, 609)
#
# # 需要删除的项目数量
# number_of_items = 1000
# time.sleep(5)
# for _ in range(number_of_items):
#     pyautogui.moveTo(first_and_second_button_coordinates[0], first_and_second_button_coordinates[1])
#     pyautogui.click()
#
#     # 等待页面反应
#     time.sleep(delay_between_clicks)
#
#     # 移动鼠标到同一位置并再次点击（第二个操作）
#     pyautogui.click()  # 坐标已经是正确的位置，无需再次移动鼠标
#
#     # 等待页面反应
#     time.sleep(delay_between_clicks)
#
#     # 移动鼠标到确认删除的按钮并点击
#     pyautogui.moveTo(confirm_delete_button_coordinates[0], confirm_delete_button_coordinates[1])
#     pyautogui.click()
#
#     # 等待页面反应
#     time.sleep(delay_between_clicks)
#
# import pyautogui
#
# input("将鼠标移动到所需位置，然后按Enter键获取坐标。")
# x, y = pyautogui.position()
# print(f"坐标：({x}, {y})")

# 坐标：(1184, 214)
#坐标：(968, 609)