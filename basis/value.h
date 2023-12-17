#ifndef VALUE_H
#define VALUE_H

#include "operation.h"

typedef struct value_ *value;

value Value(value child_left, value child_right, operation forward, operation backward);
void setData(value v, double data);
double getData(value v);
value getChildLeft(value v);
value getChildRight(value v);
void setGrad(value v, double grad);
double getGrad(value v);
void forward_value(value v);
void backward_value(value v);
void update_value(value v, double learning_rate);
void free_value(value v);

#endif // VALUE_H