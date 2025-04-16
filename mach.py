
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv.imread('./t1.jpeg')
#img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2HSV) #hsv
template = cv.imread('./t4.png',0)
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
threshold = 0.7
loc = np.where( res >= threshold)
print(min_val)
print(res >= threshold)
print(res)
print(loc)
print(min_loc)
print(min_loc[0] + w)
print(min_loc[0] + h)

for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv.imwrite('./res.png',img_rgb)

"""
import numpy as np
import matplotlib.pyplot as plt
import cv2



#读入图像，截图部分作为模板图片
img_src = cv2.imread('./t1.jpeg' ) 
img_templ = cv2.imread('./t2.png' )  
print('img_src.shape:',img_src.shape)
print('img_templ.shape:',img_templ.shape)

#模板匹配
result_t = cv2.matchTemplate(img_src, img_templ, cv2.TM_CCOEFF_NORMED)
#筛选大于一定匹配值的点
val,result = cv2.threshold(result_t,0.9,1.0,cv2.THRESH_BINARY)
match_locs = cv2.findNonZero(result)
print('match_locs.shape:',match_locs.shape) 
print('match_locs:\n',match_locs)

img_disp = img_src.copy()
for match_loc_t in match_locs:
    #match_locs是一个3维数组，第2维固定长度为1，取其下标0对应数组
    match_loc = match_loc_t[0]
    #注意计算右下角坐标时x坐标要加模板图像shape[1]表示的宽度，y坐标加高度
    right_bottom = (match_loc[0] + img_templ.shape[1], match_loc[1] + img_templ.shape[0])
    print('match_loc:',match_loc)
    print('result_t:',result_t[match_loc[1],match_loc[0]])
    #标注位置
    #cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度)
    cv2.rectangle(img_disp, match_loc, right_bottom, (0,255,0), 5)
    cv2.circle(result, match_loc, 10, (255,0,0), 3 )   
#显示图像
fig,ax = plt.subplots(2,2)
fig.suptitle('多目标匹配')
ax[0,0].set_title('img_src')
ax[0,0].imshow(cv2.cvtColor(img_src,cv2.COLOR_BGR2RGB)) 
ax[0,1].set_title('img_templ')
ax[0,1].imshow(cv2.cvtColor(img_templ,cv2.COLOR_BGR2RGB)) 
ax[1,0].set_title('result')
ax[1,0].imshow(result,'gray') 
ax[1,1].set_title('img_disp')
ax[1,1].imshow(cv2.cvtColor(img_disp,cv2.COLOR_BGR2RGB))
cv2.imwrite('./res9.png',img_disp) #rgb格式圖

#ax[0,0].axis('off');ax[0,1].axis('off');ax[1,0].axis('off');ax[1,1].axis('off')
plt.show()

