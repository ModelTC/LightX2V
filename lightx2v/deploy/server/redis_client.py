import asyncio
import json
import traceback

from loguru import logger
from redis import asyncio as aioredis

from lightx2v.deploy.common.utils import class_try_catch_async


class RedisClient:
    def __init__(self, redis_url, retry_times=3):
        self.redis_url = redis_url
        self.client = None
        self.retry_times = retry_times
        self.base_key = "lightx2v"
        self.init_scriptss()

    def init_scriptss(self):
        self.script_create_if_not_exists = """
            local key = KEYS[1]
            local data_json = ARGV[1]
            if redis.call('EXISTS', key) == 0 then
                local data = cjson.decode(data_json)
                for field, value in pairs(data) do
                    redis.call('HSET', key, field, value)
                end
                return 1
            else
                return 0
            end
        """
        self.script_increment_and_get = """
            local key = KEYS[1]
            local field = ARGV[1]
            local diff = tonumber(ARGV[2])
            local new_value = redis.call('HINCRBY', key, field, diff)
            return new_value
        """
        self.script_correct_pending_info = """
            local key = KEYS[1]
            local pending_num = tonumber(ARGV[1])
            if redis.call('EXISTS', key) ~= 0 then
                local consume_count = redis.call('HGET', key, 'consume_count')
                local max_count = redis.call('HGET', key, 'max_count')
                local redis_pending = tonumber(max_count) - tonumber(consume_count)
                if redis_pending > pending_num then
                    redis.call('HINCRBY', key, 'consume_count', redis_pending - pending_num)
                    return 'consume_count added ' .. (redis_pending - pending_num)
                end
                if redis_pending < pending_num then
                    redis.call('HINCRBY', key, 'max_count', pending_num - redis_pending)
                    return 'max_count added ' .. (pending_num - redis_pending)
                end
                return 'pending equal ' .. pending_num .. ' vs ' .. redis_pending
            else
                return 'key not exists'
            end
        """
        self.script_list_push = """
            local key = KEYS[1]
            local value = ARGV[1]
            local limit = tonumber(ARGV[2])
            redis.call('LPUSH', key, value)
            redis.call('LTRIM', key, 0, limit)
            return 1
        """
        self.script_list_avg = """
            local key = KEYS[1]
            local limit = tonumber(ARGV[1])
            local values = redis.call('LRANGE', key, 0, limit)
            local sum = 0.0
            local count = 0.0
            for _, value in ipairs(values) do
                sum = sum + tonumber(value)
                count = count + 1
            end
            if count == 0 then
                return "-1"
            end
            return tostring(sum / count)
        """

    async def init(self):
        for i in range(self.retry_times):
            try:
                self.client = aioredis.Redis.from_url(self.redis_url, protocol=3)
                ret = await self.client.ping()
                logger.info(f"Redis connection initialized, ping: {ret}")
                assert ret, "Redis connection failed"
                break
            except Exception:
                logger.warning(f"Redis connection failed, retry {i + 1}/{self.retry_times}: {traceback.format_exc()}")
                await asyncio.sleep(1)

    def fmt_key(self, key):
        return f"{self.base_key}:{key}"

    @class_try_catch_async
    async def correct_pending_info(self, key, pending_num):
        key = self.fmt_key(key)
        script = self.client.register_script(self.script_correct_pending_info)
        result = await script(keys=[key], args=[pending_num])
        logger.warning(f"Redis correct pending info {key} with {pending_num}: {result}")
        return result

    @class_try_catch_async
    async def create_if_not_exists(self, key, value):
        key = self.fmt_key(key)
        script = self.client.register_script(self.script_create_if_not_exists)
        result = await script(keys=[key], args=[json.dumps(value)])
        if result == 1:
            logger.info(f"Redis key '{key}' created successfully.")
        else:
            logger.warning(f"Redis key '{key}' already exists, not set.")

    @class_try_catch_async
    async def increment_and_get(self, key, field, diff):
        key = self.fmt_key(key)
        script = self.client.register_script(self.script_increment_and_get)
        result = await script(keys=[key], args=[field, diff])
        return result

    @class_try_catch_async
    async def hset(self, key, field, value):
        key = self.fmt_key(key)
        return await self.client.hset(key, field, value)

    @class_try_catch_async
    async def hget(self, key, field):
        key = self.fmt_key(key)
        result = await self.client.hget(key, field)
        return result

    @class_try_catch_async
    async def hgetall(self, key):
        key = self.fmt_key(key)
        result = await self.client.hgetall(key)
        return result

    @class_try_catch_async
    async def hdel(self, key, field):
        key = self.fmt_key(key)
        return await self.client.hdel(key, field)

    @class_try_catch_async
    async def hlen(self, key):
        key = self.fmt_key(key)
        result = await self.client.hlen(key)
        return result

    @class_try_catch_async
    async def set(self, key, value, nx=False):
        key = self.fmt_key(key)
        result = await self.client.set(key, value, nx=nx)
        if result is not True:
            logger.warning(f"redis set {key} = {value} failed")
        return result

    @class_try_catch_async
    async def get(self, key):
        key = self.fmt_key(key)
        result = await self.client.get(key)
        return result

    @class_try_catch_async
    async def delete_key(self, key):
        key = self.fmt_key(key)
        return await self.client.delete(key)

    @class_try_catch_async
    async def list_push(self, key, value, limit):
        key = self.fmt_key(key)
        script = self.client.register_script(self.script_list_push)
        result = await script(keys=[key], args=[value, limit])
        return result

    @class_try_catch_async
    async def list_avg(self, key, limit):
        key = self.fmt_key(key)
        script = self.client.register_script(self.script_list_avg)
        result = await script(keys=[key], args=[limit])
        return float(result)

    async def close(self):
        try:
            if self.client:
                await self.client.aclose()
            logger.info("Redis connection closed")
        except Exception:
            logger.warning(f"Error closing Redis connection: {traceback.format_exc()}")


async def main():
    redis_url = "redis://user:password@localhost:6379/1?decode_responses=True&socket_timeout=5"
    r = RedisClient(redis_url)
    await r.init()

    v1 = await r.set("test2", "1", nx=True)
    logger.info(f"set test2=1: {v1}, {await r.get('test2')}")
    v2 = await r.set("test2", "2", nx=True)
    logger.info(f"set test2=2: {v2}, {await r.get('test2')}")

    await r.create_if_not_exists("test", {"a": 1, "b": 2})
    logger.info(f"create test: {await r.hgetall('test')}")
    await r.create_if_not_exists("test", {"a": 2, "b": 3})
    logger.info(f"create test again: {await r.hgetall('test')}")
    logger.info(f"hlen test: {await r.hlen('test')}")

    v = await r.increment_and_get("test", "a", 1)
    logger.info(f"a+1: {v}, a={await r.hget('test', 'a')}")
    v = await r.increment_and_get("test", "b", 3)
    logger.info(f"b+3: {v}, b={await r.hget('test', 'b')}")

    await r.hset("test", "a", 233)
    logger.info(f"set a=233: a={await r.hget('test', 'a')}")
    await r.hset("test", "c", 456)
    logger.info(f"set c=456: c={await r.hget('test', 'c')}")
    logger.info(f"all: {await r.hgetall('test')}")
    logger.info(f"hlen test: {await r.hlen('test')}")
    logger.info(f"get unknown key: {await r.hget('test', 'unknown')}")

    await r.list_push("test_list", 1, 20)
    logger.info(f"list push 1: {await r.list_avg('test_list', 20)}")
    await r.list_push("test_list", 2, 20)
    logger.info(f"list push 2: {await r.list_avg('test_list', 20)}")
    await r.list_push("test_list", 3, 20)
    logger.info(f"list push 3: {await r.list_avg('test_list', 20)}")

    await r.delete_key("test_list")
    logger.info(f"delete test_list: {await r.list_avg('test_list', 20)}")

    await r.delete_key("test2")
    logger.info(f"delete test2: {await r.get('test2')}")

    await r.hdel("test", "a")
    logger.info(f"hdel test a: {await r.hgetall('test')}")

    await r.delete_key("test")
    logger.info(f"delete test: {await r.hgetall('test')}")
    logger.info(f"hlen test: {await r.hlen('test')}")

    await r.close()


if __name__ == "__main__":
    asyncio.run(main())
