from controller import FieldsOrder
import asyncio


set_fields = FieldsOrder()
set_fields.get_fields_from_csv()
asyncio.run(set_fields.run_in_parallel())
