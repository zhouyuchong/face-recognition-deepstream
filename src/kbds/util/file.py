import re
import os
from .constant import *

# print(os.listdir())
# path = '../../configs/lpr/config_nvdsanalytics.txt'
def init_analytics_config_file(max_source_number, type, path):
    '''
    Create an empty analytics config file. 
    + Args:
        max_source_number: to write corresponding number of groups.
    '''
    if os.path.exists(path):  # 如果文件存在
    # 删除文件，可使用以下两种方法。
        os.remove(path)
    text_line = []
    text_line.append("[property]\nenable=1\nconfig-width=1920\nconfig-height=1080\nosd-mode=2\ndisplay-font-size=12\n\n")
    file = open(path,'w')
    if type == AnalyticsValue.LINECROSSING:
        for i in range(max_source_number+1):
            text_line.append("[line-crossing-stream-{0}]\nenable=0\nline-crossing-Exit=789;672;1084;900;851;773;1203;732\nextended=0\nmode=balanced\nclass-id=2 \n\n".format(i))
    if type == AnalyticsValue.ROIREGION:
        for i in range(max_source_number+1):
            text_line.append("[roi-filtering-stream-{0}]\nroi-{1}=0;0;0;0;0;0\nenable=0\ninverse-roi=0 \nclass-id=-1 \n\n".format(i, i))
    file.writelines(text_line)
    file.close()

def modify_analytics_crossingline(filepath, max_source_number, index, enable, extended=0, mode='balanced', class_id=-1, **kwargs):
    if enable == 0:
        return True
    with open(filepath) as lf:
        text_lines = lf.readlines()    
    lf.close()

    
    comp_str = "\[line-crossing-stream-{0}\]".format(index)
    regex = re.compile(comp_str)
    for line_number, line in enumerate(text_lines):
        for match in re.findall(regex, line):
            l = line_number
            break

    if index == max_source_number:
        del text_lines[l+1:]
    else:
        comp_str_2 = "\[line-crossing-stream-{0}\]".format(index+1)
        regex_2 = re.compile(comp_str_2)
        for line_number, line in enumerate(text_lines):
            for match in re.findall(regex_2, line):
                l2 = line_number
                break
        del text_lines[l+1:l2-1]
        insert_str = ''
        if kwargs is not None:
            for key, value in kwargs.items():
                # print("{} = {}".format(key,value))
                insert_str = insert_str + 'line-crossing-{0}={1}\n'.format(key, value)
        insert_str = insert_str + 'enable={} \nextended={} \nclass-id={} \n\n'.format(enable, extended, class_id)


        text_lines.insert(l+1,insert_str)
        

    with open(filepath, 'w') as file:
        file.writelines( text_lines )

    file.close()
    return True


def modify_analytics_ROI(filepath, max_source_number, index, enable, inverse_roi=1, class_id=-1, **kwargs):
    if enable == 0:
        return True
    with open(filepath) as lf:
        text_lines = lf.readlines()    
    lf.close()

    
    comp_str = "\[roi-filtering-stream-{0}\]".format(index)
    regex = re.compile(comp_str)
    for line_number, line in enumerate(text_lines):
        for match in re.findall(regex, line):
            # print('Match found on line %d: %s' % (line_number, match))
            l = line_number
            break

    if index == max_source_number:
        del text_lines[l+1:]
    else:
        comp_str_2 = "\[roi-filtering-stream-{0}\]".format(index+1)
        regex_2 = re.compile(comp_str_2)
        for line_number, line in enumerate(text_lines):
            for match in re.findall(regex_2, line):
                # print('Match found on line %d: %s' % (line_number, match))
                l2 = line_number
                break
        del text_lines[l+1:l2-1]
        insert_str = ''
        if kwargs is not None:
            for key, value in kwargs.items():
                # print("{} = {}".format(key,value))
                insert_str = insert_str + 'roi-{0}={1}\n'.format(key, value)
        insert_str = insert_str + 'enable={} \ninverse-roi={} \nclass-id={} \n\n'.format(enable, inverse_roi, class_id)


        text_lines.insert(l+1,insert_str)
        

    with open(filepath, 'w') as file:
        file.writelines( text_lines )

    file.close()
    return True