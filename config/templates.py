# Встроенные шаблоны кампаний
# format: template_name -> {
#     "description": str,
#     "total_budget": float,
#     "example_participants": str,
#     "example_deadline": str,
#     "subtasks": [{
#         "name": str,
#         "responsible_role": str,
#         "budget_percentage": float (0-100),
#         "execution_days_ratio": float (0-1)
#     }]
# }

BUILT_IN_TEMPLATES = {
    "Таргет": {
        "description": "Кампания таргетированной рекламы на социальные сети",
        "total_budget": 100000,
        "example_participants": "Маркетолог, Таргетолог",
        "example_deadline": "25.04.2026",
        "subtasks": [
            {
                "name": "Аналитика и ЦА",
                "responsible_role": "Маркетолог",
                "budget_percentage": 10,
                "execution_days_ratio": 0.2,
            },
            {
                "name": "Креативы и тексты",
                "responsible_role": "Маркетолог",
                "budget_percentage": 20,
                "execution_days_ratio": 0.25,
            },
            {
                "name": "Настройка и запуск рекламы",
                "responsible_role": "Таргетолог",
                "budget_percentage": 50,
                "execution_days_ratio": 0.3,
            },
            {
                "name": "Оптимизация и отчетность",
                "responsible_role": "Таргетолог",
                "budget_percentage": 20,
                "execution_days_ratio": 1.0,
            },
        ],
    },
    "Контекст": {
        "description": "Контекстная реклама через Яндекс.Директ и Google Ads",
        "total_budget": 120000,
        "example_participants": "Маркетолог, PPC-специалист",
        "example_deadline": "25.04.2026",
        "subtasks": [
            {
                "name": "Сбор семантики и стратегия",
                "responsible_role": "Маркетолог",
                "budget_percentage": 16.67,
                "execution_days_ratio": 0.27,
            },
            {
                "name": "Подготовка объявлений",
                "responsible_role": "Маркетолог",
                "budget_percentage": 8.33,
                "execution_days_ratio": 0.35,
            },
            {
                "name": "Настройка и запуск кампаний",
                "responsible_role": "PPC-специалист",
                "budget_percentage": 58.33,
                "execution_days_ratio": 0.5,
            },
            {
                "name": "Оптимизация и аналитика",
                "responsible_role": "PPC-специалист",
                "budget_percentage": 16.67,
                "execution_days_ratio": 1.0,
            },
        ],
    },
    "Запуск продукта": {
        "description": "Комплексный запуск нового продукта на рынок",
        "total_budget": 200000,
        "example_participants": "Продакт-менеджер, Маркетолог, Таргетолог",
        "example_deadline": "25.04.2026",
        "subtasks": [
            {
                "name": "Исследование и стратегия продукта",
                "responsible_role": "Продакт-менеджер",
                "budget_percentage": 20,
                "execution_days_ratio": 0.27,
            },
            {
                "name": "Подготовка оффера и материалов",
                "responsible_role": "Маркетолог",
                "budget_percentage": 30,
                "execution_days_ratio": 0.6,
            },
            {
                "name": "Запуск кампании",
                "responsible_role": "Маркетолог",
                "budget_percentage": 35,
                "execution_days_ratio": 0.8,
            },
            {
                "name": "Анализ и оптимизация",
                "responsible_role": "Продакт-менеджер",
                "budget_percentage": 15,
                "execution_days_ratio": 1.0,
            },
        ],
    },
}
