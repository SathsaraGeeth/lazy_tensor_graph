#ifndef ORDER_BOOK_CJSON_H
#define ORDER_BOOK_CJSON_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cJSON {
    struct cJSON *next;
    struct cJSON *prev;
    struct cJSON *child;
    int type;
    char *valuestring;
    int valueint;
    double valuedouble;
    char *string;
} cJSON;

enum {
    cJSON_False = 0,
    cJSON_True = 1 << 1,
    cJSON_NULL = 1 << 2,
    cJSON_Number = 1 << 3,
    cJSON_String = 1 << 4,
    cJSON_Array = 1 << 5,
    cJSON_Object = 1 << 6
};

cJSON *cJSON_Parse(const char *value);
char *cJSON_PrintUnformatted(const cJSON *item);
void cJSON_Delete(cJSON *item);
cJSON *cJSON_GetArrayItem(cJSON *array, int index);
cJSON *cJSON_GetObjectItemCaseSensitive(cJSON *object, const char *string);

#define cJSON_IsNumber(item) ((item) && ((item)->type & 0xff) == cJSON_Number)
#define cJSON_IsString(item) ((item) && ((item)->type & 0xff) == cJSON_String)
#define cJSON_IsArray(item)  ((item) && ((item)->type & 0xff) == cJSON_Array)
#define cJSON_ArrayForEach(element, array) \
    for (element = (array) ? (array)->child : NULL; element; element = element->next)

#ifdef __cplusplus
}
#endif

#endif
