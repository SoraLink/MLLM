import argparse
import alibabacloud_oss_v2 as oss
import os

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="get object sample")

# 添加命令行参数 --region，表示存储空间所在的区域，必需参数
parser.add_argument('--region', help='The region in which the bucket is located.', required=True)
# 添加命令行参数 --bucket，表示存储空间的名称，必需参数
parser.add_argument('--bucket', help='The name of the bucket.', required=True)
# 添加命令行参数 --endpoint，表示其他服务可用来访问OSS的域名，非必需参数
parser.add_argument('--endpoint', help='The domain names that other services can use to access OSS')
# 添加命令行参数 --key，表示对象的名称，必需参数
parser.add_argument('--key', help='The name of the object.', required=True)

def main():
    # 解析命令行参数
    args = parser.parse_args()

    # 从环境变量中加载凭证信息，用于身份验证
    credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()

    # 加载SDK的默认配置，并设置凭证提供者
    cfg = oss.config.load_default()
    cfg.credentials_provider = credentials_provider

    # 设置配置中的区域信息
    cfg.region = args.region

    # 如果提供了endpoint参数，则设置配置中的endpoint
    if args.endpoint is not None:
        cfg.endpoint = args.endpoint

    # 使用配置好的信息创建OSS客户端
    client = oss.Client(cfg)

    # 执行获取对象的请求，指定存储空间名称和对象名称
    result = client.get_object(oss.GetObjectRequest(
        bucket=args.bucket,  # 指定存储空间名称
        key=args.key,  # 指定对象键名
    ))

    # 输出获取对象的结果信息，用于检查请求是否成功
    print(f'status code: {result.status_code},'
          f' request id: {result.request_id},'
          f' content length: {result.content_length},'
          f' content range: {result.content_range},'
          f' content type: {result.content_type},'
          f' etag: {result.etag},'
          f' last modified: {result.last_modified},'
          f' content md5: {result.content_md5},'
          f' cache control: {result.cache_control},'
          f' content disposition: {result.content_disposition},'
          f' content encoding: {result.content_encoding},'
          f' expires: {result.expires},'
          f' hash crc64: {result.hash_crc64},'
          f' storage class: {result.storage_class},'
          f' object type: {result.object_type},'
          f' version id: {result.version_id},'
          f' tagging count: {result.tagging_count},'
          f' server side encryption: {result.server_side_encryption},'
          f' server side data encryption: {result.server_side_data_encryption},'
          f' next append position: {result.next_append_position},'
          f' expiration: {result.expiration},'
          f' restore: {result.restore},'
          f' process status: {result.process_status},'
          f' delete marker: {result.delete_marker},'
    )

    # ========== 方式1：完整读取 ==========
    # with result.body as body_stream:
    #     data = body_stream.read()
    #     print(f"文件读取完成，数据长度：{len(data)} bytes")
    #
    #     path = "./get-object-sample.txt"
    #     with open(path, 'wb') as f:
    #         f.write(data)
    #     print(f"文件下载完成，保存至路径：{path}")

    # # ========== 方式2：分块读取 ==========
    with result.body as body_stream:
        chunk_path = "./get-object-sample-chunks.txt"
        total_size = 0

        with open(chunk_path, 'wb') as f:
            # 使用256KB块大小（可根据需要调整block_size参数）
            for chunk in body_stream.iter_bytes(block_size=256 * 1024):
                f.write(chunk)
                total_size += len(chunk)
                print(f"已接收数据块：{len(chunk)} bytes | 累计：{total_size} bytes")

        print(f"文件下载完成，保存至路径：{chunk_path}")

# 当此脚本被直接运行时，调用main函数
if __name__ == "__main__":
    main()  # 脚本入口，当文件被直接运行时调用main函数