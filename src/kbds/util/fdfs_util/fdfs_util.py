# coding=utf-8
import traceback
from fdfs_client.client import *


# todo------------需要完善的功能
class FastDfsUtil(object):
    """
    fast dfs 上传下载功能视图函数
    """

    def __init__(self):
        self.client_file = os.path.join(os.path.dirname(__file__), 'client.conf')
        print('self.client_file--{}'.format(self.client_file))

        self.tracker_conf = get_tracker_conf(self.client_file)
        self.client = self.create_client()

    def create_client(self):
        try:
            client = Fdfs_client(self.tracker_conf)
            return client
        except Exception as e:
            print('FastDFS客户端创建失败, {0}, {1}'.format(e, traceback.print_exc()))
            return None

    def download(self, file_name, file_id):
        """
        从Storage服务器下载文件
        :param file_name: String, 文件下载路径
        :param file_id: String, 待下载文件ID
        :return: dict {
            'Remote file_id'  : remote_file_id,
            'Content'         : local_filename,
            'Download size'   : downloaded_size,
            'Storage IP'      : storage_ip
        }
        """
        if not isinstance(file_id, bytes):
            file_id = bytes(file_id, 'UTF-8')
        try:
            ret_download = self.client.download_to_file(file_name, file_id)
            return ret_download
        except Exception as e:
            print('FastDFS文件下载失败, {0}, {1}'.format(e, traceback.print_exc()))
            return None

    def upload_by_filename(self, file_name):
        """
        上传文件到Storage服务器
        :param file_name: String, 文件上传路径
        :return: dict {
            'Group name'      : group_name,
            'Remote file_id'  : remote_file_id,
            'Status'          : 'Upload successed.',
            'Local file name' : local_file_name,
            'Uploaded size'   : upload_size,
            'Storage IP'      : storage_ip
        }
        """
        try:
            ret_upload = self.client.upload_by_filename(file_name)
            return ret_upload
        except Exception as e:
            print('FastDFS文件上传失败, {0}, {1}'.format(e, traceback.print_exc()))
            return None

    def upload_by_buffer(self, file_buffer, file_ext_name=None):
        """
        上传文件到Storage服务器
        :param file_name: String, 文件上传路径
        :return: dict {
            'Group name'      : group_name,
            'Remote file_id'  : remote_file_id,
            'Status'          : 'Upload successed.',
            'Local file name' : local_file_name,
            'Uploaded size'   : upload_size,
            'Storage IP'      : storage_ip
        }
        """
        try:
            ret_upload = self.client.upload_by_buffer(file_buffer, file_ext_name)
            return ret_upload
        except Exception as e:
            print('FastDFS文件上传失败, {0}, {1}'.format(e, traceback.print_exc()))
            return None

    def delete(self, file_id):
        """
        从Storage服务器中删除文件
        :param file_id: String, 待删除文件ID
        :return: tuple ('Delete file successed.', remote_file_id, storage_ip)
        """
        if not isinstance(file_id, bytes):
            file_id = bytes(file_id, 'UTF-8')
        try:
            ret_delete = self.client.delete_file(file_id)
            return ret_delete
        except Exception as e:
            print('FastDFS文件删除失败, {0}, {1}'.format(e, traceback.print_exc()))
            return None


if __name__ == '__main__':
    fdfs_worker = FastDfsUtil()
    file_name = os.path.join(os.path.dirname(__file__), '3.png')
    res = fdfs_worker.upload_by_filename(file_name)
    print(res)
    print('res-type{}'.format(type(res)))

    # remote_id = res['Remote file_id'] # 'group1/M00/00/00/rBuDzWLPsouAPhW_AABjMnW5uJE12.webp'
    # del_res = fdfs_worker.delete(remote_id)
    # print(del_res)

