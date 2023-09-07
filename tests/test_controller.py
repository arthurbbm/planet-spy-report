from controller import FieldsOrder
import asyncio


test = FieldsOrder()
test.get_fields_from_csv()
asyncio.run(test.run_in_parallel())
